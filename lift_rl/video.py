"""Video helpers shared by section-14/16 video scripts (copied from scripts/scripted_demo.py, which stays frozen):
close-up camera framing the cube and the target, wide camera, target wireframe projected into the image, text overlay
(phase name, label, time), real-time mp4 writer."""

from __future__ import annotations

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from .evaluation import cube_and_target
from .metrics import CUBE_REST_Z


def _t(x):
    return x.torch if hasattr(x, "torch") else x


CUBE_EDGE = 2 * CUBE_REST_Z
DIR = torch.tensor([0.34, 0.40, 0.24])
DIR = DIR / DIR.norm()
try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
except OSError:
    FONT = ImageFont.load_default(size=34)
    FONT_S = ImageFont.load_default(size=26)


def cam_cfg(focal: float, w: int = 1280, h: int = 720) -> CameraCfg:
    return CameraCfg(prim_path="{ENV_REGEX_NS}/VideoCam", update_latest_camera_pose=True, data_types=["rgb"],
                     spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.02, 20.0)),
                     width=w, height=h, renderer_cfg=NewtonWarpRendererCfg())


def aim_task(u, cam) -> None:
    """Close-up framing of the cube's start position AND the target, table surface in view."""
    cube, goal = cube_and_target(u)
    c0 = cube.clone()
    c0[:, 2] = CUBE_REST_Z
    look = 0.5 * (c0 + goal) + torch.tensor([0.0, 0.0, 0.07], device=u.device)  # the hand sits above the cube
    dist = (0.6 + 1.6 * torch.linalg.norm(goal - c0, dim=-1)).clamp(0.8, 1.7)
    o = u.scene.env_origins
    cam.set_world_poses_from_view(o + look + dist[:, None] * DIR.to(u.device), o + look)


WIDE_EYE, WIDE_LOOK = (1.35, 1.00, 0.85), (0.35, 0.0, 0.25)  # farther and higher than lift_rl.camera's EYE/TARGET


def aim_wide(u, cam) -> None:
    o = u.scene.env_origins
    cam.set_world_poses_from_view(o + torch.tensor(WIDE_EYE, device=u.device), o + torch.tensor(WIDE_LOOK, device=u.device))


def grab(u, name: str) -> np.ndarray:
    rgb = _t(u.scene[name].data.output["rgb"])
    return rgb[..., :3].to(torch.uint8).cpu().numpy()


def project(u, name: str, pts_env: torch.Tensor) -> np.ndarray:
    """(N,K,3) env-frame points -> (N,K,3) pixel u, v, depth for camera `name`."""
    cam = u.scene[name]
    pos = _t(cam.data.pos_w)
    Rw = matrix_from_quat(_t(cam.data.quat_w_ros))  # ROS: +z forward, +x right, +y down
    K = _t(cam.data.intrinsic_matrices)
    pw = pts_env + u.scene.env_origins[:, None]
    pc = torch.einsum("nji,nkj->nki", Rw, pw - pos[:, None])
    uvw = torch.einsum("nij,nkj->nki", K, pc)
    return torch.cat([uvw[..., :2] / uvw[..., 2:3].clamp_min(1e-6), pc[..., 2:3]], -1).cpu().numpy()


_CORNERS = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32) * (CUBE_EDGE / 2)
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if int((_CORNERS[a] != _CORNERS[b]).sum()) == 1]


def target_points(u) -> torch.Tensor:
    _, goal = cube_and_target(u)
    pts = goal[:, None] + _CORNERS.to(u.device)[None]
    floor = goal.clone()
    floor[:, 2] = 0.0
    return torch.cat([pts, goal[:, None], floor[:, None]], 1)  # 8 corners, center, center projected to the table


def overlay(frame: np.ndarray, uvd: np.ndarray, lines: list[str], hit: bool) -> np.ndarray:
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im, "RGBA")
    col = (40, 230, 90, 255) if hit else (255, 60, 200, 255)
    if (uvd[:, 2] > 0).all():
        for k in range(0, 1000, 2):  # dashed drop line from the target to the table
            a, b = uvd[8, :2], uvd[9, :2]
            s0, s1 = k / 40, (k + 1) / 40
            if s0 >= 1:
                break
            dr.line([tuple(a + (b - a) * s0), tuple(a + (b - a) * min(s1, 1))], fill=col[:3] + (150,), width=2)
        for a, b in _EDGES:
            dr.line([tuple(uvd[a, :2]), tuple(uvd[b, :2])], fill=col, width=4)
        cx, cy = uvd[8, :2]
        dr.text((cx + 26, cy - 40), "target", font=FONT_S, fill=col)
    y = 14
    for j, s in enumerate(lines):
        f = FONT if j == 0 else FONT_S
        bb = dr.textbbox((18, y), s, font=f)
        dr.rectangle([bb[0] - 8, bb[1] - 6, bb[2] + 8, bb[3] + 6], fill=(0, 0, 0, 150))
        dr.text((18, y), s, font=f, fill=(255, 255, 255, 255))
        y = bb[3] + 14
    return np.asarray(im)


def writer(path: Path, fps: int = 50):
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=8)


