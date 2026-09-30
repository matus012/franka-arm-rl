"""lift_rl: Franka cube lift with PPO on Isaac Lab 3.0 + Newton. Importing registers the gym tasks."""

import gymnasium as gym

_ENTRY = "lift_rl.guarded_env:GuardedRLEnv"
_STOCK_PPO = "isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"

# M0: stock task, Newton physics, NaN guard.
gym.register(
    id="LiftRL-Stock-Newton-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.stock_cfg:StockLiftNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": _STOCK_PPO,
    },
    disable_env_checker=True,
)

# M0 physics-config fix attempt 1: stock MDP, finger armature + stiffer contacts.
gym.register(
    id="LiftRL-Stock-Newton-Fix1-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.stock_cfg:StockLiftNewtonFixEnvCfg",
        "rsl_rl_cfg_entry_point": _STOCK_PPO,
    },
    disable_env_checker=True,
)

# M0 physics-config fix attempt 2: fix 1 + 4 substeps + solref 0.005 s.
gym.register(
    id="LiftRL-Stock-Newton-Fix2-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.stock_cfg:StockLiftNewtonFix2EnvCfg",
        "rsl_rl_cfg_entry_point": _STOCK_PPO,
    },
    disable_env_checker=True,
)

# M1: own env + own reward.
gym.register(
    id="LiftRL-Own-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.env_cfg:LiftEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# M2: M1 + slam/jam penalties + mass/friction randomization.
gym.register(
    id="LiftRL-Twist-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.env_cfg:LiftTwistEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Run 2 (brief 10.2): staged pick reward, fix-2 physics; and its domain-randomized twist arm.
gym.register(
    id="LiftRL-Staged-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.staged_cfg:LiftStagedEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Staged-DR-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.staged_cfg:LiftStagedDREnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Section 11 probe 5: staged reward + scripted close trigger (gripper close not learned).
gym.register(
    id="LiftRL-Staged-AutoClose-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.staged_cfg:LiftStagedAutoCloseEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Section 12 frozen setup (probe 5 + fixed lr 5e-5): baseline and domain-randomized twist arm.
gym.register(
    id="LiftRL-Frozen-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.staged_cfg:LiftStagedAutoCloseEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPOFixedLRRunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Frozen-DR-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.staged_cfg:LiftStagedAutoCloseDREnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPOFixedLRRunnerCfg",
    },
    disable_env_checker=True,
)

# M0 diagnostic: stock IK-abs action variant on Newton, driven by the scripted grasp.
gym.register(
    id="LiftRL-Stock-IK-Newton-v0",
    entry_point=_ENTRY,
    kwargs={"env_cfg_entry_point": "lift_rl.stock_cfg:StockLiftIKNewtonEnvCfg"},
    disable_env_checker=True,
)

# Section 16 (step 1): phase machine + gripper reflex, one fixed world, 8 s episodes.
gym.register(
    id="LiftRL-Phased-Fixed-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedFixedEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P3-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP3EnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P4-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP4EnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P5-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP5EnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P6-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP6EnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-Fixed-TS-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedFixedTSEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P3-TS-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP3TSEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P4-TS-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP4TSEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P5-TS-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP5TSEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P6-TS-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP6TSEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-Fixed-TS-Demo-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedFixedTSDemoEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P3-TS-Demo-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP3TSDemoEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P4-TS-Demo-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP4TSDemoEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P5-TS-Demo-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP5TSDemoEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="LiftRL-Phased-P6-TS-Demo-v0",
    entry_point=_ENTRY,
    kwargs={
        "env_cfg_entry_point": "lift_rl.phased_cfg:LiftPhasedP6TSDemoEnvCfg",
        "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPhasedPPORunnerCfg",
    },
    disable_env_checker=True,
)
