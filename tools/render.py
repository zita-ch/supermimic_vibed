"""Render recorded evaluation trajectories (supermimic/evaluate.py --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance1_subject1 jumps1_subject1 --seconds 12
"""

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
# MuJoCo renders with Mesa's software rasterizer by default. SM_RENDER_GPU=1 selects the NVIDIA EGL
# vendor instead; that path hung after a few frames in the kit-less container and is not validated.
_NV_EGL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker", "10_nvidia.json")
if os.environ.get("SM_RENDER_GPU") == "1" and os.path.exists(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", os.path.abspath(_NV_EGL))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MJCF = os.path.join(ROOT, "data", "g1_description", "g1_29dof.xml")


def two_robot_xml() -> str:
    xml = open(MJCF).read()
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="' + os.path.join(ROOT, 'data', 'g1_description') + '/"', xml)
    body = xml[xml.index('<body name="pelvis"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.35 0.65 1.0 0.25"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<mujoco", '<mujoco', 1)
    if "<visual>" not in xml:
        xml = xml.replace("<asset>", '<visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--global_ghost", action="store_true", help="draw the reference at its global xy instead of anchored to the robot")
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    names = list(d["names"])
    model = mujoco.MjModel.from_xml_string(two_robot_xml())
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135.0, -12.0
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    clips = a.clips or names
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = int(round(50 / a.fps))
    for clip in clips:
        c = names.index(clip)
        T = d["root_p"].shape[0]
        alive = d["alive"][:, c]
        t0 = int(a.start * 50)
        t1 = min(T, t0 + int(a.seconds * 50))
        fell_at = None
        for t in range(t0, t1, stride):
            if not alive[t] and fell_at is None:
                fell_at = t
            tt = t if alive[t] else (fell_at if fell_at is not None else t)
            f = int(d["f"][tt, c])
            data.qpos[0:3] = d["root_p"][tt, c]
            data.qpos[3:7] = d["root_quat"][tt, c]
            data.qpos[7:36] = d["q"][tt, c]
            data.qpos[36:39] = d["ref_root_p"][f]
            if not a.global_ghost:  # anchor the reference to the robot's xy, as the tracking reward does
                data.qpos[36:38] = d["root_p"][tt, c][:2]
            data.qpos[39:43] = _mat_to_quat(d["ref_root_R"][f])
            data.qpos[43:72] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            label = f"{a.title}  {clip}  t={(f - int(d['clip_start'][c])) * 0.02:5.1f}s"
            if fell_at is not None:
                label += "  [terminated]"
            for xy, txt in (((16, 12), label), ((16, a.height - 38), "solid: simulated G1 policy (Isaac Lab + MuJoCo-Warp)   blue ghost: reference")):
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


def _mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


if __name__ == "__main__":
    main()
