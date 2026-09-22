"""Exact near-optimal landscape Omega_eps (scoring-only object; never given to an arm)."""
from __future__ import annotations

import networkx as nx


def all_independent_sets(G: nx.Graph) -> list[int]:
    """Every independent set of G as a bitmask (vertices 0..n-1). Exact enumeration."""
    n = G.number_of_nodes()
    nb = [0] * n
    for u, v in G.edges():
        nb[u] |= 1 << v
        nb[v] |= 1 << u
    out = []

    def rec(i: int, mask: int, forbidden: int):
        if i == n:
            out.append(mask)
            return
        rec(i + 1, mask, forbidden)                       # exclude i
        if not (forbidden >> i) & 1:                      # include i
            rec(i + 1, mask | (1 << i), forbidden | nb[i])
    rec(0, 0, 0)
    return out


def weight(mask: int, w: dict) -> float:
    return sum(w[i] for i in w if (mask >> i) & 1)


def omega_eps(G, w, grid, min_size=100):
    """Apply the frozen epsilon rule. Returns (eps_rel, E_star, Omega list, flagged)."""
    sets = all_independent_sets(G)
    W = {m: weight(m, w) for m in sets}
    Wmax = max(W.values())
    E_star = -Wmax
    for eps in grid:
        omega = [m for m in sets if -W[m] <= E_star + eps * abs(E_star) + 1e-12]
        if len(omega) >= min_size:
            return eps, E_star, omega, False
    return grid[-1], E_star, omega, True


def is_maximal(mask: int, G: nx.Graph) -> bool:
    n = G.number_of_nodes()
    for v in range(n):
        if (mask >> v) & 1:
            continue
        if not any((mask >> u) & 1 for u in G.adj[v]):
            return False
    return True


def swap_components(omega: list[int]) -> list[list[int]]:
    """Components of x ~ y iff |x| = |y| and |x symdiff y| = 2 (single swap)."""
    members = set(omega)
    n = max(omega).bit_length() if omega else 0
    n = max(n, 1)
    comp, seen = [], set()
    for start in omega:
        if start in seen:
            continue
        stack, cur = [start], []
        seen.add(start)
        while stack:
            x = stack.pop()
            cur.append(x)
            ins = [i for i in range(n + 32) if (x >> i) & 1]
            for u in ins:
                base = x & ~(1 << u)
                for v in range(n + 32):
                    if (x >> v) & 1 or v == u:
                        continue
                    y = base | (1 << v)
                    if y in members and y not in seen:
                        seen.add(y)
                        stack.append(y)
        comp.append(cur)
    return comp


def jaccard(a: int, b: int) -> float:
    """d_J on bitmask sets; d_J(empty, empty) = 0 by the frozen convention."""
    union = bin(a | b).count("1")
    if union == 0:
        return 0.0
    return 1.0 - bin(a & b).count("1") / union


def coverage_radius(omega: list[int]) -> float:
    """r(G): median over y of the Jaccard distance to its nearest other member of Omega."""
    import statistics
    if len(omega) < 2:
        return 0.0
    nn = [min(jaccard(y, z) for z in omega if z != y) for y in omega]
    return statistics.median(nn)
