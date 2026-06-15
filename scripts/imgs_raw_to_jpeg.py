#!/usr/bin/env python3
"""
imgs_raw_to_jpeg.py — 把已打包的 X-VLA h5（图像是裸像素 (T,H,W,3)）原地重编码为
JPEG vlen-uint8（每帧一条 1D 字节流），对齐 X-VLA datasets.utils.decode_image_from_bytes
（用 cv2.imdecode 解码）。不重新渲染，仅重编码现有像素。

用法：python imgs_raw_to_jpeg.py --root <xvla_hdf5 目录> [--quality 95]
"""
import argparse, glob, os, shutil, tempfile
import numpy as np, h5py, cv2

IMG_KEYS = ["observations/agentview_image", "observations/robot0_eye_in_hand_image"]


def jpeg_encode_frames(frames, quality):
    param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    out = []
    for fr in frames:
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(fr), param)
        if not ok:
            raise RuntimeError("cv2.imencode 失败")
        out.append(np.frombuffer(buf.tobytes(), dtype=np.uint8))
    return out


def is_already_jpeg(ds):
    # vlen 数据集 dtype 是 object/特殊；裸像素是 3D uint8
    return ds.ndim == 1


def convert_file(fp, quality):
    with h5py.File(fp, "r") as f:
        # 已是 vlen（1D）就跳过
        if all(is_already_jpeg(f[k]) for k in IMG_KEYS):
            return "skip"
        abs6d = f["abs_action_6d"][()]
        lang = f["language_instruction"][()]
        imgs = {k: f[k][()] for k in IMG_KEYS}
        attrs = dict(f.attrs)

    tmp = fp + ".tmp"
    with h5py.File(tmp, "w") as o:
        o.create_dataset("abs_action_6d", data=abs6d)
        g = o.create_group("observations")
        for k in IMG_KEYS:
            name = k.split("/")[-1]
            jpegs = jpeg_encode_frames(imgs[k], quality)
            ds = g.create_dataset(name, (len(jpegs),), dtype=h5py.vlen_dtype(np.uint8))
            for i, b in enumerate(jpegs):
                ds[i] = b
        o.create_dataset("language_instruction", data=lang)
        for ak, av in attrs.items():
            o.attrs[ak] = av
    os.replace(tmp, fp)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, "**", "*.h5"), recursive=True))
    print(f"待转换 {len(files)} 个 h5")
    n_ok = n_skip = 0
    for i, fp in enumerate(files):
        r = convert_file(fp, args.quality)
        n_ok += (r == "ok"); n_skip += (r == "skip")
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"  [{i+1}/{len(files)}] ok={n_ok} skip={n_skip}")
    print(f"完成：重编码 {n_ok}，跳过(已是jpeg) {n_skip}")


if __name__ == "__main__":
    main()
