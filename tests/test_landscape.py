import itertools

import networkx as nx

from certisample.instances import make_instance
from certisample.landscape import all_independent_sets, is_maximal, swap_components


def brute(G):
    n = G.number_of_nodes()
    out = []
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            if not any(G.has_edge(u, v) for u, v in itertools.combinations(S, 2)):
                out.append(sum(1 << i for i in S))
    return sorted(out)


def test_enumeration_matches_bruteforce():
    for seed in range(5):
        G = nx.gnp_random_graph(12, 0.35, seed=seed)
        assert sorted(all_independent_sets(G)) == brute(G)


def test_generator_is_deterministic_and_blockade():
    a, b = make_instance(7, "pilot"), make_instance(7, "pilot")
    assert a.positions == b.positions and a.w == b.w
    assert max(a.w.values()) == 1.0 and len(a.positions) == 20
    for u, v in a.G.edges():
        (x1, y1), (x2, y2) = a.positions[u], a.positions[v]
        assert max(abs(x1 - x2), abs(y1 - y2)) <= 1


def test_swap_components_and_maximal():
    G = nx.path_graph(4)                  # sets {0,2},{0,3},{1,3} are one swap-connected class
    sets = [0b0101, 0b1001, 0b1010]
    comps = swap_components(sets)
    assert len(comps) == 1
    assert is_maximal(0b0101, G) and not is_maximal(0b0001, G)


def test_jaccard_and_radius():
    from certisample.landscape import coverage_radius, jaccard
    assert jaccard(0, 0) == 0.0
    assert jaccard(0b011, 0b011) == 0.0
    assert abs(jaccard(0b011, 0b110) - (1 - 1 / 3)) < 1e-12
    assert coverage_radius([0b01, 0b11]) == 0.5
