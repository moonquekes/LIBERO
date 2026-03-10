import mujoco
import numpy as np


class SuctionStickyWrapper:
    def __init__(self, env, suction_threshold=0.0):
        self.env = env
        self.suction_threshold = suction_threshold
        self.attached_body_id = None
        self.rel_pos = None
        self.rel_mat = None
        self._cache_ready = False
        self._suction_geom_ids = set()
        self._robot_body_ids = set()
        self._grip_site_id = None
        self._indicator_site_id = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        self._refresh_cache()
        self._detach()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._refresh_cache()
        suction_on = float(action[-1]) > self.suction_threshold

        if suction_on:
            if self.attached_body_id is None:
                self._try_attach_from_contacts()
            if self.attached_body_id is not None:
                self._update_attached_pose()
        else:
            self._detach()

        # 更新吸盘状态指示颜色：绿=开，红=关
        self._update_suction_indicator(suction_on)

        self.env.sim.forward()
        return obs, reward, done, info

    def _refresh_cache(self):
        if self._cache_ready:
            return
        model = self.env.sim.model

        for geom_id in range(model.ngeom):
            name = model.geom_id2name(geom_id)
            if name and name.endswith("suction_pad_collision"):
                self._suction_geom_ids.add(geom_id)

        for body_id in range(model.nbody):
            body_name = model.body_id2name(body_id)
            if body_name and body_name.startswith("robot0_"):
                self._robot_body_ids.add(body_id)

        for site_id in range(model.nsite):
            site_name = model.site_id2name(site_id)
            if site_name and site_name.endswith("grip_site"):
                self._grip_site_id = site_id
                break

        # 吸盘开关指示 site
        for site_id in range(model.nsite):
            site_name = model.site_id2name(site_id)
            if site_name and site_name.endswith("suction_indicator"):
                self._indicator_site_id = site_id
                break

        self._cache_ready = True

    def _body_has_free_joint(self, body_id):
        model = self.env.sim.model
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]
        if jnt_num <= 0:
            return False
        jnt_type = model.jnt_type[jnt_adr]
        return int(jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)

    def _try_attach_from_contacts(self):
        if self._grip_site_id is None or not self._suction_geom_ids:
            return

        model = self.env.sim.model
        data = self.env.sim.data
        for i in range(data.ncon):
            con = data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            if g1 in self._suction_geom_ids:
                other_geom = g2
            elif g2 in self._suction_geom_ids:
                other_geom = g1
            else:
                continue

            body_id = int(model.geom_bodyid[other_geom])
            if body_id in self._robot_body_ids:
                continue
            if not self._body_has_free_joint(body_id):
                continue

            self.attached_body_id = body_id
            grip_pos = data.site_xpos[self._grip_site_id].copy()
            grip_mat = data.site_xmat[self._grip_site_id].reshape(3, 3).copy()
            body_pos = data.xpos[body_id].copy()
            body_mat = data.xmat[body_id].reshape(3, 3).copy()
            self.rel_pos = grip_mat.T @ (body_pos - grip_pos)
            self.rel_mat = grip_mat.T @ body_mat
            return

    def _update_attached_pose(self):
        body_id = self.attached_body_id
        if body_id is None:
            return

        model = self.env.sim.model
        data = self.env.sim.data

        grip_pos = data.site_xpos[self._grip_site_id]
        grip_mat = data.site_xmat[self._grip_site_id].reshape(3, 3)
        body_pos = grip_pos + grip_mat @ self.rel_pos
        body_mat = grip_mat @ self.rel_mat

        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, body_mat.reshape(-1))

        jnt_adr = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_adr]
        qvel_adr = model.jnt_dofadr[jnt_adr]

        data.qpos[qpos_adr : qpos_adr + 3] = body_pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _detach(self):
        self.attached_body_id = None
        self.rel_pos = None
        self.rel_mat = None

    def _update_suction_indicator(self, suction_on: bool):
        """把 suction_indicator site 改为绿色（开）或红色（关）。只改 R/G，不动 alpha。"""
        if self._indicator_site_id is None:
            return
        rgba = self.env.sim.model.site_rgba[self._indicator_site_id]
        if suction_on:
            rgba[0] = 0.0  # R
            rgba[1] = 1.0  # G
            rgba[2] = 0.0  # B
        else:
            rgba[0] = 1.0  # R
            rgba[1] = 0.0  # G
            rgba[2] = 0.0  # B
        # alpha 不在这里修改：由外部 set_suction_indicator_visibility 控制
