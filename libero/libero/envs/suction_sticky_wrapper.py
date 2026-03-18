import mujoco
import numpy as np


class SuctionStickyWrapper:
    REASON_CODES = {
        "off": 0,
        "attached_ok": 1,
        "no_contact": 2,
        "invalid_body": 3,
        "angle_exceeded": 4,
        "outside_effective_radius": 5,
        "contact_lost": 6,
    }
    REASON_LABELS = {
        0: "off",
        1: "attached_ok",
        2: "no_contact",
        3: "invalid_body",
        4: "angle_exceeded",
        5: "outside_effective_radius",
        6: "contact_lost",
    }

    def __init__(
        self,
        env,
        suction_threshold=0.0,
        normal_max_angle_deg=25.0,
        effective_radius_ratio=0.7,
        detach_grace_steps=2,
        record_diagnostics=True,
    ):
        self.env = env
        self.suction_threshold = suction_threshold
        self.normal_max_angle_deg = float(normal_max_angle_deg)
        self.effective_radius_ratio = max(float(effective_radius_ratio), 0.0)
        self.detach_grace_steps = max(int(detach_grace_steps), 0)
        self.record_diagnostics = bool(record_diagnostics)
        self.attached_body_id = None
        self.rel_pos = None
        self.rel_mat = None
        self._cache_ready = False
        self._robot_body_ids = set()
        self._grip_site_id = None
        self._suction_site_id = None
        self._indicator_site_id = None
        self._pad_geom_id = None
        self._pad_radius = 0.0
        self._normal_alignment_threshold = float(
            np.cos(np.deg2rad(self.normal_max_angle_deg))
        )
        self._invalid_contact_steps = 0
        self._attached_candidate = None
        self._suction_on = False
        self._diagnostics = self._make_diagnostics(
            reason_code=self.REASON_CODES["off"],
            attached=False,
            constraint_ok=False,
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        self._refresh_cache()
        self._detach()
        self._suction_on = False
        self._set_diagnostics(
            reason_code=self.REASON_CODES["off"],
            attached=False,
            constraint_ok=False,
        )
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._refresh_cache()
        suction_on = float(action[-1]) > self.suction_threshold
        self._suction_on = bool(suction_on)

        if not suction_on:
            self._detach()
            self._set_diagnostics(
                reason_code=self.REASON_CODES["off"],
                attached=False,
                constraint_ok=False,
            )
        elif self.attached_body_id is None:
            candidate = self._select_best_candidate()
            if candidate is not None and candidate["constraint_ok"]:
                self._attach_candidate(candidate)
                self._update_attached_pose()
                self._set_diagnostics(
                    reason_code=self.REASON_CODES["attached_ok"],
                    attached=True,
                    constraint_ok=True,
                    candidate=candidate,
                )
            else:
                self._invalid_contact_steps = 0
                self._set_diagnostics_from_candidate(
                    candidate,
                    fallback_reason=self.REASON_CODES["no_contact"],
                    attached=False,
                    constraint_ok=False,
                )
        else:
            candidate = self._select_best_candidate(attached_body_id=self.attached_body_id)
            if candidate is not None and candidate["constraint_ok"]:
                self._attached_candidate = dict(candidate)
            self._invalid_contact_steps = 0
            self._update_attached_pose()
            self._set_diagnostics(
                reason_code=self.REASON_CODES["attached_ok"],
                attached=True,
                constraint_ok=True,
                candidate=self._attached_candidate,
            )

        self._update_suction_indicator(suction_on)
        self.env.sim.forward()
        return obs, reward, done, info

    def get_suction_diagnostics(self):
        return dict(self._diagnostics)

    @classmethod
    def get_suction_reason_codes(cls):
        return dict(cls.REASON_CODES)

    def is_suction_on(self):
        return bool(self._suction_on)

    def get_current_constraint_probe(self):
        self._refresh_cache()
        attached_body_id = self.attached_body_id if self.attached_body_id is not None else None
        candidate = self._select_best_candidate(attached_body_id=attached_body_id)
        if candidate is None:
            return {
                "available": False,
                "constraint_ok": False,
                "reason_code": self.REASON_CODES["no_contact"],
                "reason_label": self.REASON_LABELS[self.REASON_CODES["no_contact"]],
                "contact_angle_deg": np.nan,
                "contact_radial_offset_m": np.nan,
                "contact_body_id": -1,
            }
        return {
            "available": True,
            "constraint_ok": bool(candidate["constraint_ok"]),
            "reason_code": int(candidate["reason_code"]),
            "reason_label": self.REASON_LABELS.get(int(candidate["reason_code"]), "unknown"),
            "contact_angle_deg": float(candidate["angle_deg"]),
            "contact_radial_offset_m": float(candidate["radial_offset"]),
            "contact_body_id": int(candidate["body_id"]),
        }

    def get_nearest_attachable_distance(self):
        self._refresh_cache()
        if self.attached_body_id is not None:
            return 0.0, int(self.attached_body_id)
        if self._suction_site_id is None:
            return np.inf, -1

        model = self.env.sim.model
        data = self.env.sim.data
        suction_center = np.asarray(data.site_xpos[self._suction_site_id], dtype=np.float64)
        best_distance = np.inf
        best_body_id = -1

        for geom_id in range(model.ngeom):
            if geom_id == self._pad_geom_id:
                continue
            body_id = int(model.geom_bodyid[geom_id])
            if body_id in self._robot_body_ids:
                continue
            if not self._body_has_free_joint(body_id):
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
                continue

            distance = self._point_to_geom_surface_distance(suction_center, geom_id)
            if distance < best_distance:
                best_distance = distance
                best_body_id = body_id

        return float(best_distance), int(best_body_id)

    def _make_diagnostics(self, reason_code, attached, constraint_ok, candidate=None):
        contact_angle_deg = np.nan
        contact_radial_offset_m = np.nan
        contact_body_id = -1
        if candidate is not None:
            contact_angle_deg = float(candidate["angle_deg"])
            contact_radial_offset_m = float(candidate["radial_offset"])
            contact_body_id = int(candidate["body_id"])

        return {
            "constraint_ok": bool(constraint_ok),
            "attached": bool(attached),
            "reason_code": int(reason_code),
            "reason_label": self.REASON_LABELS.get(int(reason_code), "unknown"),
            "contact_angle_deg": contact_angle_deg,
            "contact_radial_offset_m": contact_radial_offset_m,
            "contact_body_id": contact_body_id,
        }

    def _set_diagnostics(self, reason_code, attached, constraint_ok, candidate=None):
        self._diagnostics = self._make_diagnostics(
            reason_code=reason_code,
            attached=attached,
            constraint_ok=constraint_ok,
            candidate=candidate,
        )

    def _set_diagnostics_from_candidate(
        self,
        candidate,
        fallback_reason,
        attached,
        constraint_ok,
    ):
        if candidate is None:
            self._set_diagnostics(
                reason_code=fallback_reason,
                attached=attached,
                constraint_ok=constraint_ok,
            )
            return

        self._set_diagnostics(
            reason_code=int(candidate["reason_code"]),
            attached=attached,
            constraint_ok=constraint_ok,
            candidate=candidate,
        )

    def _refresh_cache(self):
        if self._cache_ready:
            return
        model = self.env.sim.model

        for geom_id in range(model.ngeom):
            name = model.geom_id2name(geom_id)
            if name and name.endswith("suction_pad_collision"):
                self._pad_geom_id = geom_id
                self._pad_radius = float(model.geom_size[geom_id][0])
                break

        for body_id in range(model.nbody):
            body_name = model.body_id2name(body_id)
            if body_name and body_name.startswith("robot0_"):
                self._robot_body_ids.add(body_id)

        for site_id in range(model.nsite):
            site_name = model.site_id2name(site_id)
            if site_name and site_name.endswith("grip_site"):
                self._grip_site_id = site_id
            if site_name and site_name.endswith("suction_site"):
                self._suction_site_id = site_id
            if site_name and site_name.endswith("suction_indicator"):
                self._indicator_site_id = site_id

        self._cache_ready = True

    def _body_has_free_joint(self, body_id):
        model = self.env.sim.model
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]
        if jnt_num <= 0:
            return False
        jnt_type = model.jnt_type[jnt_adr]
        return int(jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)

    def _get_pad_axis(self):
        if self._suction_site_id is None:
            return None
        suction_mat = self.env.sim.data.site_xmat[self._suction_site_id].reshape(3, 3)
        pad_axis = suction_mat[:, 2].astype(np.float64)
        norm = np.linalg.norm(pad_axis)
        if norm == 0:
            return None
        return pad_axis / norm

    def _point_to_geom_surface_distance(self, point, geom_id):
        model = self.env.sim.model
        data = self.env.sim.data

        geom_type = int(model.geom_type[geom_id])
        geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        local_point = geom_mat.T @ (np.asarray(point, dtype=np.float64) - geom_pos)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            q = np.abs(local_point) - geom_size
            outside = np.linalg.norm(np.maximum(q, 0.0))
            inside = min(float(np.max(q)), 0.0)
            return max(float(outside + inside), 0.0)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return max(float(np.linalg.norm(local_point) - geom_size[0]), 0.0)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radial = np.linalg.norm(local_point[:2])
            dr = radial - geom_size[0]
            dz = abs(float(local_point[2])) - geom_size[1]
            outside = np.linalg.norm(np.maximum(np.array([dr, dz]), 0.0))
            inside = min(max(float(dr), float(dz)), 0.0)
            return max(float(outside + inside), 0.0)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half_length = float(geom_size[1])
            closest = np.array([0.0, 0.0, np.clip(local_point[2], -half_length, half_length)])
            return max(float(np.linalg.norm(local_point - closest) - geom_size[0]), 0.0)

        return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - geom_pos))

    def _estimate_support_region(
        self,
        other_geom,
        suction_center,
        contact_normal,
        effective_radius,
    ):
        model = self.env.sim.model
        data = self.env.sim.data

        geom_type = int(model.geom_type[other_geom])
        geom_size = np.asarray(model.geom_size[other_geom], dtype=np.float64)
        geom_pos = np.asarray(data.geom_xpos[other_geom], dtype=np.float64)
        geom_mat = np.asarray(data.geom_xmat[other_geom], dtype=np.float64).reshape(3, 3)

        local_center = geom_mat.T @ (suction_center - geom_pos)
        local_normal = geom_mat.T @ contact_normal
        dominant_axis = int(np.argmax(np.abs(local_normal)))
        tangential_axes = [axis for axis in range(3) if axis != dominant_axis]
        tangential_center = local_center[tangential_axes]
        radial_offset = float(np.linalg.norm(tangential_center))
        support_tolerance = 1e-4

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            face_half_extents = geom_size[tangential_axes]
            face_margins = face_half_extents - np.abs(tangential_center)
            support_ok = bool(np.all(face_margins >= (effective_radius - support_tolerance)))
            return radial_offset, support_ok

        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            if dominant_axis != 2:
                return radial_offset, False
            face_radius = float(geom_size[0])
            support_ok = radial_offset <= max(face_radius - effective_radius + support_tolerance, 0.0)
            return radial_offset, bool(support_ok)

        return np.nan, None

    def _collect_contact_candidates(self, attached_body_id=None):
        if self._pad_geom_id is None or self._suction_site_id is None:
            return []

        model = self.env.sim.model
        data = self.env.sim.data
        pad_axis = self._get_pad_axis()
        if pad_axis is None:
            return []

        suction_center = data.site_xpos[self._suction_site_id].copy()
        effective_radius = self._pad_radius * self.effective_radius_ratio
        candidates = []
        for i in range(data.ncon):
            con = data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            if g1 == self._pad_geom_id:
                other_geom = g2
            elif g2 == self._pad_geom_id:
                other_geom = g1
            else:
                continue

            body_id = int(model.geom_bodyid[other_geom])
            if body_id in self._robot_body_ids:
                continue
            if attached_body_id is not None and body_id != attached_body_id:
                continue

            contact_normal = np.asarray(con.frame, dtype=np.float64)[:3]
            normal_norm = np.linalg.norm(contact_normal)
            if normal_norm == 0:
                continue
            contact_normal /= normal_norm

            alignment = float(np.clip(np.abs(np.dot(pad_axis, contact_normal)), 0.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(alignment)))

            radial_offset, support_ok = self._estimate_support_region(
                other_geom=other_geom,
                suction_center=suction_center,
                contact_normal=contact_normal,
                effective_radius=effective_radius,
            )

            reason_code = self.REASON_CODES["attached_ok"]
            constraint_ok = True
            if not self._body_has_free_joint(body_id):
                reason_code = self.REASON_CODES["invalid_body"]
                constraint_ok = False
            elif alignment < self._normal_alignment_threshold:
                reason_code = self.REASON_CODES["angle_exceeded"]
                constraint_ok = False
            elif support_ok is False:
                reason_code = self.REASON_CODES["outside_effective_radius"]
                constraint_ok = False

            candidates.append(
                {
                    "body_id": body_id,
                    "reason_code": reason_code,
                    "constraint_ok": constraint_ok,
                    "alignment": alignment,
                    "angle_deg": angle_deg,
                    "radial_offset": radial_offset,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["radial_offset"],
                item["angle_deg"],
                item["body_id"],
            )
        )
        return candidates

    def _select_best_candidate(self, attached_body_id=None):
        candidates = self._collect_contact_candidates(attached_body_id=attached_body_id)
        if not candidates:
            return None

        for candidate in candidates:
            if candidate["constraint_ok"]:
                return candidate
        return candidates[0]

    def _attach_candidate(self, candidate):
        if self._grip_site_id is None:
            return
        body_id = int(candidate["body_id"])
        data = self.env.sim.data
        self.attached_body_id = body_id
        grip_pos = data.site_xpos[self._grip_site_id].copy()
        grip_mat = data.site_xmat[self._grip_site_id].reshape(3, 3).copy()
        body_pos = data.xpos[body_id].copy()
        body_mat = data.xmat[body_id].reshape(3, 3).copy()
        self.rel_pos = grip_mat.T @ (body_pos - grip_pos)
        self.rel_mat = grip_mat.T @ body_mat
        self._invalid_contact_steps = 0
        self._attached_candidate = dict(candidate)

    def _update_attached_pose(self):
        body_id = self.attached_body_id
        if (
            body_id is None
            or self.rel_pos is None
            or self.rel_mat is None
            or self._grip_site_id is None
        ):
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
        self._invalid_contact_steps = 0
        self._attached_candidate = None

    def _update_suction_indicator(self, suction_on: bool):
        if self._indicator_site_id is None:
            return
        rgba = self.env.sim.model.site_rgba[self._indicator_site_id]
        if not suction_on:
            rgba[0] = 1.0
            rgba[1] = 0.0
            rgba[2] = 0.0
        elif self._diagnostics["attached"] and self._diagnostics["constraint_ok"]:
            rgba[0] = 0.0
            rgba[1] = 1.0
            rgba[2] = 0.0
        else:
            rgba[0] = 1.0
            rgba[1] = 1.0
            rgba[2] = 0.0
        # alpha is controlled by the external visualization helpers
