"""v8 count-unit ablation over the frozen v7 phase-masked selector.

Only the observation's available shadow-history count changes units. A scale of
20 encodes the same causal information as a fraction of the 20-session window;
a scale of 1 is an exact observation control. No learned scaler or clipping is
used. Trading, rewards, shadow histories, reset schedules and masks are inherited.
"""
from __future__ import annotations

import numpy as np

from .masked_selector_v7 import MaskedSelectorEnvV7


class ScaledSelectorEnvV8(MaskedSelectorEnvV7):
    """Divide the single count feature by 1 or 20; all else stays unchanged."""

    def __init__(self, panel, start, end, training=False, episode_length=126,
                 rebalance_every=5, cost_bps=5, actions=None,
                 drawdown_penalty=True, *, count_scale=20):
        if (isinstance(count_scale, (bool, np.bool_))
                or not isinstance(count_scale, (int, float, np.integer, np.floating))
                or count_scale not in (1, 20)):
            raise ValueError("count_scale must be 1 or 20")
        # Set before super: the inherited constructor calls reset/_observation.
        self.count_scale = float(count_scale)
        super().__init__(panel, start, end, training=training,
                         episode_length=episode_length,
                         rebalance_every=rebalance_every, cost_bps=cost_bps,
                         actions=actions, drawdown_penalty=drawdown_penalty,
                         mask_mode="phase")

    @property
    def count_index(self):
        # Base: 9*N+2; four cash-first targets: 4*(N+1); performance: 12.
        return 13 * self.base.n_assets + 18

    def _observation(self):
        observation = super()._observation()
        observation[self.count_index] /= np.float32(self.count_scale)
        return observation
