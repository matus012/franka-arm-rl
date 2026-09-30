"""PPO (rsl_rl) runner config for the own env. Starts identical to the stock LiftCubePPORunnerCfg
(Isaac Lab, BSD-3-Clause): 24 steps/env, 1,500 iterations, MLP 256-128-64 ELU, lr 1e-4 adaptive (KL 0.01),
gamma 0.98, lambda 0.95, entropy 0.006, 5 epochs x 4 minibatches."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50  # run 2 step 3 scores checkpoints 1,000 / 1,250 / 1,500
    experiment_name = "lift_rl"
    seed = 42
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiftPPOFixedLRRunnerCfg(LiftPPORunnerCfg):
    """Section 12 frozen setup: identical PPO, except a fixed learning rate of 5e-5 (no adaptive KL schedule).
    Probe 5 collapsed after iteration ~260 under the adaptive schedule."""

    def __post_init__(self):
        self.algorithm.schedule = "fixed"
        self.algorithm.learning_rate = 5.0e-5


@configclass
class MinStdGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = "lift_rl.rl_ext:MinStdGaussianDistribution"
    min_std: float = 0.05
    max_std: float | None = None


@configclass
class LiftPhasedPPORunnerCfg(LiftPPORunnerCfg):
    """Section 16.5: stock PPO, gamma 0.99 (8 s episodes), adaptive lr 1e-4, checkpoint every 50.
    Section 17.3c: entropy coefficient 0.01 and a minimum action std (exploration collapsed in probe 1)."""

    def __post_init__(self):
        self.algorithm.gamma = 0.99
        self.algorithm.entropy_coef = 0.01
        # min std 0.05, not 0.2: with 0.2 action noise even the scripted arm never reaches CLOSE (0 %), with 0.1 67 %, with
        # 0.05 100 % (results/phased_proof_clean_noise*.json); probe 2 (min std 0.2) never reached CLOSE
        # 17.3d: initial std 0.5 (run 3's first random walks learned "stay away from the table")
        self.actor.distribution_cfg = MinStdGaussianDistributionCfg(init_std=0.5, min_std=0.05, max_std=0.6)
        # run 5: entropy 0.01 -> 0.005 and std capped at 0.6; run 4's std climbed 0.27 -> 1.17 on the
        # plateau (saturated 1 cm/step jitter in task space)
        self.algorithm.entropy_coef = 0.005
