"""Video camera for replay/video scripts only. Never added to a training env.

Newton cameras only see their own env, so a multi-episode grid is N per-env views tiled together
(gotcha 4a.6). ``update_latest_camera_pose=True`` so a moved camera does not freeze at its reset pose.
"""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.sensors import CameraCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils

EYE = (1.10, 0.80, 0.62)  # relative to env origin [m]
TARGET = (0.32, 0.0, 0.22)


def add_video_camera(cfg, width: int = 480, height: int = 360) -> None:
    cfg.scene.video_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCam",
        update_latest_camera_pose=True,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
        ),
        width=width,
        height=height,
        renderer_cfg=NewtonWarpRendererCfg(),
    )


CLOSEUP_OFFSET = (0.34, 0.40, 0.24)  # eye relative to the cube's start position: ~0.58 m away, looking down
CLOSEUP_LOOK = (0.0, 0.0, 0.06)


def add_closeup_camera(cfg, width: int = 1280, height: int = 720) -> None:
    """Close-up mode (brief 10.3): 1280x720, ~0.6 m from the grasp, table surface in view."""
    cfg.scene.video_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCam",
        update_latest_camera_pose=True,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.02, 20.0)
        ),
        width=width,
        height=height,
        renderer_cfg=NewtonWarpRendererCfg(),
    )


def aim_closeup(env) -> None:
    """Point each env's camera at its cube's current (start) position."""
    u = env.unwrapped
    c = u.scene["object"].data.root_pos_w
    c = c.torch if hasattr(c, "torch") else c
    u.scene["video_cam"].set_world_poses_from_view(
        c + torch.tensor(CLOSEUP_OFFSET, device=u.device), c + torch.tensor(CLOSEUP_LOOK, device=u.device)
    )


class SlowMoWriter:
    """Streams one env's frames (one per 50 Hz control step) to an mp4 at 25 fps: the first `slow_steps`
    steps (2 s) are written twice each -> 0.25x speed, the rest every 2nd step -> 1x."""

    def __init__(self, path, slow_steps: int = 100, keep_steps: tuple[int, ...] = ()):
        import imageio

        self.w = imageio.get_writer(str(path), fps=25, codec="libx264", quality=8, macro_block_size=8)
        self.slow, self.t, self.keep, self.kept = slow_steps, 0, set(keep_steps), {}

    def add(self, frame: np.ndarray) -> None:
        if self.t < self.slow:
            self.w.append_data(frame)
            self.w.append_data(frame)
        elif (self.t - self.slow) % 2 == 0:
            self.w.append_data(frame)
        if self.t in self.keep:
            self.kept[self.t] = frame.copy()
        self.t += 1

    def close(self) -> None:
        self.w.close()


def aim_video_camera(env, eye=EYE, target=TARGET) -> None:
    u = env.unwrapped
    cam = u.scene["video_cam"]
    origins = u.scene.env_origins
    eyes = origins + torch.tensor(eye, device=u.device)
    targets = origins + torch.tensor(target, device=u.device)
    cam.set_world_poses_from_view(eyes, targets)


def grab_rgb(env) -> np.ndarray:
    """(N, H, W, 3) uint8 frames, one per env."""
    rgb = env.unwrapped.scene["video_cam"].data.output["rgb"]
    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
    return rgb[..., :3].to(torch.uint8).cpu().numpy()


def tile(frames: np.ndarray, cols: int) -> np.ndarray:
    n, h, w, c = frames.shape
    rows = (n + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(n):
        r, k = divmod(i, cols)
        out[r * h : (r + 1) * h, k * w : (k + 1) * w] = frames[i]
    return out
