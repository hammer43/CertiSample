import networkx as nx
import numpy as np

from certisample.instances import make_instance
from certisample.samplers import ARMS, Problem
from certisample.scoring import Scorer, paired_bootstrap_lb

PARAMS = {"randomized_greedy_ls": dict(tie_band=0.1, ls_rounds=20),
          "constraint_aware_gibbs": dict(beta=4, sweeps=50, burn_in_sweeps=20),
          "simulated_annealing_is": dict(beta_final=8, sweeps=50)}


def independent(p, m):
    return all(not (p.nb[v] & m) for v in range(p.n) if (m >> v) & 1)


def test_all_samplers_return_K_independent_sets():
    inst = make_instance(3, "pilot")
    p = Problem(inst.G, inst.w)
    for name, fn in ARMS.items():
        outs, evals = fn(p, 7, np.random.default_rng(0), **PARAMS[name])
        assert len(outs) == 7 and evals > 0
        assert all(independent(p, m) for m in outs), name


def test_samplers_are_seed_deterministic():
    inst = make_instance(4, "pilot")
    p = Problem(inst.G, inst.w)
    for name, fn in ARMS.items():
        a = fn(p, 5, np.random.default_rng(11), **PARAMS[name])[0]
        b = fn(p, 5, np.random.default_rng(11), **PARAMS[name])[0]
        assert a == b, name


def test_local_search_never_reduces_weight():
    from certisample.samplers import _greedy, _local_search
    inst = make_instance(5, "pilot")
    p = Problem(inst.G, inst.w)
    rng = np.random.default_rng(1)
    for _ in range(10):
        c = [0]
        g = _greedy(p, 0.2, rng, c)
        ls = _local_search(p, g, 50, rng, c)
        assert p.weight(ls) >= p.weight(g) - 1e-12 and independent(p, ls)


def test_scorer_known_values():
    omega = [0b0011, 0b0110, 0b1100]
    sc = Scorer(omega, radius=0.5, components=[omega])
    full = sc.score(omega, lambda m: True)
    assert full["mean_nearest_jaccard"] == 0.0 and full["radius_coverage_fraction"] == 1.0
    empty = sc.score([0b10000], lambda m: True)
    assert empty["mean_nearest_jaccard"] == 1.0 and empty["radius_coverage_fraction"] == 0.0
    assert empty["out_of_window_fraction"] == 1.0


def test_bootstrap_lb():
    assert paired_bootstrap_lb([1.0] * 30, 1000, 0) == 1.0
    assert paired_bootstrap_lb(np.r_[np.ones(15), -np.ones(15)], 2000, 0) < 0
