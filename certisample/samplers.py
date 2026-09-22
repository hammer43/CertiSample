"""Omega-blind classical samplers for A-G0 (prereg/g0_spec.yaml, arms.classical).

Each sampler returns K raw proposals (bitmasks over vertices 0..n-1) and a count of
elementary energy evaluations. None of them ever sees Omega_eps, E* or the landscape.

Implementation details the frozen spec leaves open, fixed here and logged in every
selection file:
  * Gibbs: each proposal is an independent chain from the empty set, run for
    burn_in_sweeps + sweeps, returning its final state.
  * Annealing: each proposal is an independent run from the empty set with a geometric
    beta schedule from BETA0 = 0.1 to beta_final over `sweeps`, returning its final state.
  * A sweep is n single-site heat-bath updates at uniformly random sites.
"""
from __future__ import annotations

import math

BETA0 = 0.1


class Problem:
    """Bitmask view of a weighted unit-disk graph."""

    def __init__(self, G, w):
        self.n = G.number_of_nodes()
        self.nb = [0] * self.n
        for u, v in G.edges():
            self.nb[u] |= 1 << v
            self.nb[v] |= 1 << u
        self.w = [w[i] for i in range(self.n)]

    def weight(self, m: int) -> float:
        return sum(self.w[i] for i in range(self.n) if (m >> i) & 1)


# ---------------------------------------------------------------- greedy + local search
def _greedy(p: Problem, tie_band: float, rng, counter):
    avail, I = (1 << p.n) - 1, 0
    while avail:
        verts = [v for v in range(p.n) if (avail >> v) & 1]
        score = {v: p.w[v] / (1 + bin(p.nb[v] & avail).count("1")) for v in verts}
        counter[0] += len(verts)
        top = max(score.values())
        cand = [v for v in verts if score[v] >= top * (1 - tie_band) - 1e-15]
        v = cand[int(rng.integers(len(cand)))]
        I |= 1 << v
        avail &= ~((1 << v) | p.nb[v])
    return I


def _local_search(p: Problem, I: int, rounds: int, rng, counter):
    """(1,2)-swap local search for MWIS; single free insertions are tried first."""
    for _ in range(rounds):
        moved = False
        free = [u for u in range(p.n) if not (I >> u) & 1 and not (p.nb[u] & I)]
        counter[0] += p.n
        if free:
            I |= 1 << free[int(rng.integers(len(free)))]
            continue
        ins = [v for v in range(p.n) if (I >> v) & 1]
        for k in rng.permutation(len(ins)):
            v = ins[int(k)]
            rest = I & ~(1 << v)
            F = [u for u in range(p.n) if not (rest >> u) & 1 and u != v and not (p.nb[u] & rest)]
            counter[0] += len(F)
            best = None
            for i, a in enumerate(F):
                for b in F[i + 1:]:
                    if not (p.nb[a] >> b) & 1:
                        gain = p.w[a] + p.w[b] - p.w[v]
                        if gain > 1e-12 and (best is None or gain > best[0]):
                            best = (gain, a, b)
            if best:
                I = rest | (1 << best[1]) | (1 << best[2])
                moved = True
                break
        if not moved:
            break
    return I


def randomized_greedy_ls(p: Problem, K: int, rng, tie_band: float, ls_rounds: int):
    counter = [0]
    out = [_local_search(p, _greedy(p, tie_band, rng, counter), ls_rounds, rng, counter)
           for _ in range(K)]
    return out, counter[0]


# ---------------------------------------------------------------- heat-bath kernels
def _sweep(p: Problem, I: int, beta: float, rng, counter):
    sites = rng.integers(0, p.n, p.n)
    u = rng.random(p.n)
    for s, r in zip(sites.tolist(), u.tolist()):
        counter[0] += 1
        x = beta * p.w[s]
        p_on = 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0
        if (I >> s) & 1:
            if r > p_on:
                I &= ~(1 << s)
        elif not (p.nb[s] & I) and r < p_on:
            I |= 1 << s
    return I


def constraint_aware_gibbs(p: Problem, K: int, rng, beta: float, sweeps: int, burn_in_sweeps: int):
    counter, out = [0], []
    for _ in range(K):
        I = 0
        for _ in range(burn_in_sweeps + sweeps):
            I = _sweep(p, I, beta, rng, counter)
        out.append(I)
    return out, counter[0]


def simulated_annealing_is(p: Problem, K: int, rng, beta_final: float, sweeps: int):
    counter, out = [0], []
    ratio = (beta_final / BETA0) ** (1.0 / max(1, sweeps - 1))
    for _ in range(K):
        I, beta = 0, BETA0
        for _ in range(sweeps):
            I = _sweep(p, I, beta, rng, counter)
            beta *= ratio
        out.append(I)
    return out, counter[0]


ARMS = {
    "randomized_greedy_ls": randomized_greedy_ls,
    "constraint_aware_gibbs": constraint_aware_gibbs,
    "simulated_annealing_is": simulated_annealing_is,
}
IMPLEMENTATION_NOTES = {
    "gibbs": "independent chain per proposal from the empty set; burn_in_sweeps + sweeps; final state",
    "annealing": f"independent run per proposal from the empty set; geometric beta {BETA0} -> beta_final",
    "sweep": "n single-site heat-bath updates at uniformly random sites",
    "local_search": "free insertion first, else best weight-improving (1,2)-swap; stop when none",
}
