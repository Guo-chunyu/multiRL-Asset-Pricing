"""v7 causal action-mask ablation over the frozen v6 daily selector.

Masking ignored requests may remove nuisance policy gradients; it is a research
hypothesis, not an established explanation of policy collapse. Both arms must
use the same official sb3_contrib MaskablePPO backend and explicit prediction
masks. This module changes neither the daily ledger nor either v6 objective.
"""
from __future__ import annotations

import numpy as np

from .selector_env_v6 import FixedExpertSelectorEnv


class MaskedSelectorEnvV7(FixedExpertSelectorEnv):
    """Expose pre-step phase masks without changing observations or transitions.

    ``phase`` permits all experts on rebalance decisions and only index zero
    otherwise. Index zero is an ignored request on these non-rebalance days,
    NOT an instruction to liquidate or a request queued for the next rebalance.
    ``control`` permits all four experts every day. The mask is solely a
    function of the current calendar phase, never future prices or returns.

    Like v6, ``step`` accepts any valid expert index even when masked out:
    masks constrain policy sampling, not the accounting transition. This makes
    identical executed choices directly comparable across the two arms.
    """

    def __init__(self, panel, start, end, training=False, episode_length=126,
                 rebalance_every=5, cost_bps=5, actions=None,
                 drawdown_penalty=True, *, mask_mode="phase"):
        if mask_mode not in ("phase", "control"):
            raise ValueError("mask_mode must be 'phase' or 'control'")
        self.mask_mode = mask_mode
        self.training_episode_schedule = []
        super().__init__(panel, start, end, training=training,
                         episode_length=episode_length,
                         rebalance_every=rebalance_every, cost_bps=cost_bps,
                         actions=actions, drawdown_penalty=drawdown_penalty)

    def reset(self, *, seed=None, options=None):
        """Trace sampled fragments without altering the inherited reset.

        A seeded reset starts a new trace, discarding the constructor reset;
        subsequent unseeded vector-env resets append to that same run.
        """
        observation, info = super().reset(seed=seed, options=options)
        if seed is not None:
            self.training_episode_schedule.clear()
        if self.base.training:
            self.training_episode_schedule.append((self.base.start, self.base.end))
        return observation, info

    def action_masks(self):
        """Return a fresh boolean mask for the CURRENT, pre-step decision."""
        if self.mask_mode == "control" or self.rebalance_phase == 0:
            return np.ones(self.action_space.n, dtype=np.bool_)
        return np.array([True, False, False, False], dtype=np.bool_)
