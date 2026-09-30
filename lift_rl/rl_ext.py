"""rsl_rl 5 extension (section 17.3c/d): Gaussian action distribution with a minimum (and optional maximum) std.

The exploration noise of the section-16 probe collapsed (mean std 1.0 -> 0.075 by iteration 300). The std parameter
is clamped from below in the forward pass, so the sampled noise never drops under MIN_STD; the deterministic
(mean) policy used for evaluation is unchanged. rsl_rl logs the clamped value as Policy/mean_std.
"""

from __future__ import annotations

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal


class MinStdGaussianDistribution(GaussianDistribution):
    def __init__(self, output_dim: int, init_std: float = 1.0, std_type: str = "scalar", min_std: float = 0.2,
                 max_std: float | None = None):
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        self.min_std, self.max_std = min_std, max_std

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.expand_as(mean)
        else:
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = Normal(mean, std.clamp(self.min_std, self.max_std))
