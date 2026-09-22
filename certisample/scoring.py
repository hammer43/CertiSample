"""A-G0 scoring (prereg/g0_spec.yaml, metrics). Uses Omega_eps as a scoring oracle only."""
from __future__ import annotations

import numpy as np

from .landscape import jaccard


class Scorer:
    def __init__(self, omega: list[int], radius: float, components: list[list[int]]):
        self.omega = omega
        self.omega_set = set(omega)
        self.r = radius
        self.comp_of = {m: i for i, c in enumerate(components) for m in c}
        self.n_comp = len(components)

    def score(self, outputs: list[int], is_feasible) -> dict:
        K = len(outputs)
        feas = [o for o in outputs if is_feasible(o)]
        uniq = set(outputs)
        S = [m for m in uniq if m in self.omega_set]
        if S:
            nn = [min(jaccard(x, y) for x in S) for y in self.omega]
            mean_nearest = float(np.mean(nn))
            radius_cov = float(np.mean([d <= self.r + 1e-12 for d in nn]))
            worst = float(max(nn))
            comp_cov = len({self.comp_of[m] for m in S}) / self.n_comp
        else:
            mean_nearest, radius_cov, worst, comp_cov = 1.0, 0.0, 1.0, 0.0
        return dict(
            mean_nearest_jaccard=mean_nearest,            # co-primary A (lower is better)
            radius_coverage_fraction=radius_cov,          # co-primary B (higher is better)
            worst_case_jaccard_radius=worst,
            swap_component_coverage=comp_cov,
            unique_in_window=len(S),
            infeasible_fraction=1 - len(feas) / K,
            out_of_window_fraction=sum(1 for o in feas if o not in self.omega_set) / K,
            duplicate_fraction=1 - len(uniq) / K,
        )


def paired_bootstrap_lb(diffs, resamples: int, seed: int, interval: float = 0.95) -> float:
    """Lower bound of the percentile bootstrap CI of the mean paired difference."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (resamples, len(d)))].mean(1)
    return float(np.quantile(means, (1 - interval) / 2))
