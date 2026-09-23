"""Regression tests for the EV application model (spec v1.4.1)."""
import math

import networkx as nx
import numpy as np
import pytest

from certisample.ev import expanded, recourse, tntp
from certisample.ev.instance import RHO_GRID, load_network, make_instance

DATA = __import__("os").environ.get("CERTISAMPLE_TNTP", "data/tntp")
pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(f"{DATA}/SiouxFalls_net.tntp").exists(),
    reason="TNTP data not present")


@pytest.fixture(scope="module")
def net():
    return load_network("SiouxFalls", f"{DATA}/SiouxFalls_net.tntp",
                        f"{DATA}/SiouxFalls_trips.tntp", f"{DATA}/SiouxFalls_node.tntp")


def test_network_and_R_EV(net):
    assert len(net.nodes) == 24 and net.G.number_of_edges() == 76
    d = sorted(net.sp[o][dd] for (o, dd), f in net.trips.items()
               if o != dd and f > 0 and dd in net.sp[o])
    assert abs(net.R_EV - float(np.median(d))) < 1e-12          # spec 2.3
    assert all(net.sp[o][dd] > net.R_EV for o, dd in net.K_plus)


def test_instance_is_seed_reproducible(net):
    a, b = make_instance(net, 3, 100), make_instance(net, 3, 100)
    assert a.sites == b.sites and a.od == b.od and a.demand == b.demand
    c = make_instance(net, 4, 100)
    assert (a.sites, a.od) != (c.sites, c.od)


def test_candidate_and_demand_rules(net):
    I = make_instance(net, 0, 100)
    assert len(I.sites) == math.floor(0.80 * 24) == 19          # spec 3.2
    assert len(I.od) == 100 and set(I.od) <= set(net.K_plus)    # spec 3.3
    assert abs(np.mean(list(I.demand.values())) - 1.0) < 1e-12  # spec 3.4


def test_conflict_graph_matches_definition(net):
    I = make_instance(net, 1, 100)
    c = net.coords_norm
    for rho in RHO_GRID:
        H = I.conflict_graph(rho)
        assert set(H.nodes()) == set(I.sites)
        for i in I.sites:
            for j in I.sites:
                if i < j:
                    assert H.has_edge(i, j) == (math.dist(c[i], c[j]) < rho)   # strict
    sizes = [I.conflict_graph(r).number_of_edges() for r in RHO_GRID]
    assert sizes == sorted(sizes)                               # monotone in rho


def test_conflict_graph_independent_of_phi_and_instance_pairing(net):
    I = make_instance(net, 2, 100)
    assert I.conflict_graph(1.25).edges() == make_instance(net, 2, 100).conflict_graph(1.25).edges()


def test_opening_costs(net):
    I = make_instance(net, 0, 100)
    for phi in (1.0, 10.0, 100.0):
        f = I.opening_costs(phi)
        raw = {i: phi for i in I.sites}
        for (o, d), dk in I.demand.items():
            for node in (o, d):
                if node in raw:
                    raw[node] -= 0.5 * dk
        assert f == raw                                          # spec 6 and 8
    assert all(v > 0 for v in I.opening_costs(100.0).values())   # high phi: all positive


def test_recourse_integrality_and_paths(net):
    I = make_instance(net, 0, 40)
    E = expanded.build(I)
    y = {i: 1 for i in I.sites}
    r = recourse.evaluate(I, E, y, with_duals=False)
    assert r["feasible"]
    for k, path in r["paths"].items():
        assert path[0] == ("O", k) and path[-1] == ("D", k)      # integral path solution
        arcs = {(t, h) for t, h, _ in E.arcs[k]}
        assert all((a, b) in arcs for a, b in zip(path, path[1:]))


def test_empty_y_is_infeasible_and_ray_is_normalized(net):
    I = make_instance(net, 0, 40)
    E = expanded.build(I)
    r = recourse.evaluate(I, E, {i: 0 for i in I.sites})
    assert not r["feasible"] and r["delta"]
    nrm = math.sqrt(r["gamma"] ** 2 + sum(v * v for v in r["delta"].values()))
    assert abs(nrm - 1.0) < 1e-12                                # spec 12A.3
    assert r["gamma"] > 0 and all(v < 0 for v in r["delta"].values())


def test_cut_is_tight_at_generator_and_valid_elsewhere(net):
    I = make_instance(net, 0, 30)
    E = expanded.build(I)
    rng = np.random.default_rng(0)
    y = None
    for _ in range(40):
        cand = {i: int(v) for i, v in zip(I.sites, rng.integers(0, 2, len(I.sites)))}
        r = recourse.evaluate(I, E, cand)
        if r["feasible"]:
            y = cand
            break
    assert y is not None
    lhs = r["alpha"] + sum(r["beta"][j] * y[j] for j in I.sites)
    assert abs(lhs - r["Q"]) < 1e-6 * len(I.od)                  # tight at generator
                                                                 # (dual-face slack per pair)
    for _ in range(6):
        y2 = {i: int(v) for i, v in zip(I.sites, rng.integers(0, 2, len(I.sites)))}
        r2 = recourse.evaluate(I, E, y2, with_duals=False)
        if r2["feasible"]:
            lhs2 = r["alpha"] + sum(r["beta"][j] * y2[j] for j in I.sites)
            assert lhs2 <= r2["Q"] + 1e-6                        # valid elsewhere


def test_duals_are_deterministic(net):
    I = make_instance(net, 0, 20)
    E = expanded.build(I)
    y = {i: 1 if k % 2 == 0 else 0 for k, i in enumerate(I.sites)}
    r1 = recourse.evaluate(I, E, y)
    r2 = recourse.evaluate(I, E, y)
    if r1["feasible"]:
        assert abs(r1["alpha"] - r2["alpha"]) < 1e-12
        assert all(abs(r1["beta"][j] - r2["beta"][j]) < 1e-12 for j in r1["beta"])


def test_master_lexicographic_tie_break(net):
    highspy = pytest.importorskip("highspy")
    from certisample.ev.master import solve_master
    sites = [1, 2, 3, 4]
    fbar = {i: 0.0 for i in sites}                # every y is primary-optimal
    y, obj, th, status = solve_master(sites, fbar, [], use_theta=False)
    assert "Optimal" in status and y == {i: 0 for i in sites}
    # {3} and {4} tie on the primary objective; lexicographic order minimizes the
    # earlier site first, so the tie-break selects y3 = 0 and y4 = 1.
    fbar2 = {1: 0.0, 2: 0.0, 3: -1.0, 4: -1.0}
    y2, obj2, *_ = solve_master(sites, fbar2, [(3, 4)], use_theta=False)
    assert obj2 == -1.0 and y2[3] == 0 and y2[4] == 1
