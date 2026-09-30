# Third-party software and assets

| component | licence | how it is used here |
|---|---|---|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0.0b2.post1 (tag `v3.0.0-beta2.patch1`) | BSD-3-Clause | pip dependency. Files copied and modified in this repo (headers kept, changes marked `lift_rl:`): `scripts/train.py`, `scripts/play.py`, `scripts/cli_args.py`; `ManagerBasedRLEnv.step` in `lift_rl/guarded_env.py`; the lift task config in `lift_rl/env_cfg.py`; the stack task's Newton physics preset in `lift_rl/physics.py`; the PPO config in `lift_rl/agents.py`. |
| [Newton](https://github.com/newton-physics/newton) 1.2.1 | Apache-2.0 | physics engine (pip dependency) |
| [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) 3.8.1 / MuJoCo 3.8.1 | Apache-2.0 | contact solver used by Newton (pip dependency) |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) 1.13.0 | Apache-2.0 | GPU kernels (pip dependency) |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (rsl-rl-lib 5.0.1) | BSD-3-Clause | PPO implementation (pip dependency); `lift_rl/rl_ext.py` subclasses its `GaussianDistribution` (std clamp) |
| PyTorch 2.10.0+cu130 | BSD-3-Clause | pip dependency |
| NVIDIA assets: Franka Panda (`Robots/FrankaEmika/Legacy/panda_instanceable.usd`), SeattleLabTable, DexCube | NVIDIA asset terms | **loaded at runtime** from NVIDIA's content server; **not redistributed** (no asset file is in this repo). Rendered frames of them appear in the videos. |

Our own code (everything not listed above as copied from Isaac Lab) is MIT-licensed (see LICENSE in the published repository).
