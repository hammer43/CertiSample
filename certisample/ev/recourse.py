"""Recourse LP, deterministic duals and feasibility cuts (spec sections 7, 9, 12A.2-12A.4).

For fixed y the recourse decomposes by OD pair: each pair is a shortest-path LP on its
expanded arcs with the activation constraint sum_{i:(i,j)} x_ijk <= y_j at physical nodes.

Optimality-cut coefficients use the deterministic dual of spec 12A.2: solve the primal,
restrict to the optimal dual face, fix the gauge at the artificial destination, then
minimize the L1 norm of the dual vector (LP epigraph). Summing over pairs gives
alpha = sum_k (lambda_Oa_k - lambda_Da_k) and beta_j = sum_k mu_{j,k} (spec section 9).
"""
from __future__ import annotations

import heapq
import itertools
import math

import highspy
import numpy as np

TOL = 1e-9
FACE_TOL = 1e-10
LP_KKT_TOL = 1e-10


def nominal_face_budget(n_pairs: int, face_tol: float = FACE_TOL) -> float:
    """Nominal aggregate optimal-face drift budget: |K| * tau_face (spec v1.4.2)."""
    return n_pairs * face_tol



def _adj(arcs, open_nodes):
    adj = {}
    for t, h, ell in arcs:
        if (isinstance(t, int) and t not in open_nodes) or (isinstance(h, int) and h not in open_nodes):
            continue
        adj.setdefault(t, []).append((h, ell))
    return adj


def shortest(arcs, source, target, open_nodes):
    adj = _adj(arcs, open_nodes)
    counter = itertools.count()
    dist, prev = {source: 0.0}, {}
    pq = [(0.0, next(counter), source)]
    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf) + TOL:
            continue
        for v, ell in sorted(adj.get(u, []), key=lambda e: (e[1], str(e[0]))):
            nd = d + ell
            if nd < dist.get(v, math.inf) - TOL:
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, next(counter), v))
    return dist, prev


def reachable(arcs, source, open_nodes) -> set:
    adj = _adj(arcs, open_nodes | {source} if isinstance(source, int) else open_nodes)
    seen, stack = {source}, [source]
    while stack:
        u = stack.pop()
        for v, _ in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _configure(h):
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("random_seed", 0)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    # FACE_TOL is 1e-10, so the LP/KKT tolerances must not be looser than it.
    # HiGHS defaults are 1e-7 and would make the face budget meaningless.
    h.setOptionValue("kkt_tolerance", LP_KKT_TOL)
    h.setOptionValue("primal_feasibility_tolerance", LP_KKT_TOL)
    h.setOptionValue("dual_feasibility_tolerance", LP_KKT_TOL)
    h.setOptionValue("primal_residual_tolerance", LP_KKT_TOL)
    h.setOptionValue("dual_residual_tolerance", LP_KKT_TOL)
    h.setOptionValue("optimality_tolerance", LP_KKT_TOL)


def dual_for_pair(arcs, src, dst, phys, y, cost_scale, Qk, face_tol: float = FACE_TOL):
    """Deterministic optimal dual (spec 12A.2) for one OD pair.

    max  lambda_src - lambda_dst + sum_j y_j nu_j
    s.t. lambda_t - lambda_h + [h physical] nu_h <= cost_scale * ell   for every arc
         nu_j <= 0
    then: objective fixed to Qk, gauge lambda_dst = 0, minimize ||(lambda, nu)||_1.
    """
    nodes = sorted({t for t, _, _ in arcs} | {h for _, h, _ in arcs}, key=str)
    idx = {n: i for i, n in enumerate(nodes)}
    pj = {j: len(nodes) + i for i, j in enumerate(phys)}
    n = len(nodes) + len(phys)

    h = highspy.Highs()
    _configure(h)
    lam = [h.addVariable(lb=-highspy.kHighsInf, ub=highspy.kHighsInf) for _ in nodes]
    nu = [h.addVariable(lb=-highspy.kHighsInf, ub=0.0) for _ in phys]
    u = [h.addVariable(lb=0.0, ub=highspy.kHighsInf) for _ in range(n)]   # L1 epigraph
    var = lam + nu

    for t, hd, ell in arcs:
        e = lam[idx[t]] - lam[idx[hd]]
        if isinstance(hd, int):
            e = e + nu[phys.index(hd)]
        h.addConstr(e <= cost_scale * ell)
    obj = lam[idx[src]] - lam[idx[dst]]
    for j in phys:
        if y.get(j, 0) > 0.5:
            obj = obj + nu[phys.index(j)]
    h.addConstr(obj >= Qk - face_tol)      # optimal dual face (spec 12A.2 step 2)
    h.addConstr(obj <= Qk + face_tol)
    h.addConstr(lam[idx[dst]] == 0.0)      # gauge
    for i in range(n):
        h.addConstr(var[i] <= u[i])
        h.addConstr(-var[i] <= u[i])
    h.minimize(sum(u))                     # minimum L1 on the optimal face
    status = h.modelStatusToString(h.getModelStatus())
    if "Optimal" not in status:
        raise RuntimeError(f"dual selection failed: {status}")
    lam_v = {nd: float(h.val(lam[idx[nd]])) for nd in nodes}
    nu_v = {j: float(h.val(nu[phys.index(j)])) for j in phys}
    return lam_v, nu_v


def evaluate(inst, exp, y: dict, with_duals: bool = True) -> dict:
    """Q(y) and the Benders cut for this candidate (spec sections 7 and 9)."""
    open_nodes = {i for i, v in y.items() if v > 0.5}
    Q, paths = 0.0, {}
    alpha, beta = 0.0, {j: 0.0 for j in exp.physical}
    for k in inst.od:
        src, dst = ("O", k), ("D", k)
        dist, prev = shortest(exp.arcs[k], src, dst, open_nodes)
        if dst not in dist:                                   # infeasible recourse
            S = reachable(exp.arcs[k], src, open_nodes)
            frontier = sorted({hd for t, hd, _ in exp.arcs[k]
                               if t in S and isinstance(hd, int) and hd not in open_nodes})
            if not frontier:
                return dict(feasible=False, unroutable=k, gamma=None, delta=None)
            g, d = 1.0, {j: -1.0 for j in frontier}          # 1 - sum_F y_j <= 0
            nrm = math.sqrt(g ** 2 + sum(v * v for v in d.values()))
            return dict(feasible=False, unroutable=k, gamma=g / nrm,
                        delta={j: v / nrm for j, v in d.items()})
        dk = inst.demand[k]
        Q += dk * dist[dst]
        node, path = dst, []
        while node != src:
            path.append(node)
            node = prev[node]
        paths[k] = [src] + path[::-1]
        if with_duals:
            lam, nu = dual_for_pair(exp.arcs[k], src, dst, exp.physical, y, dk, dk * dist[dst])
            alpha += lam[src] - lam[dst]
            for j in exp.physical:
                beta[j] += nu[j]
    face_lhs = alpha + sum(beta[j] * y.get(j, 0) for j in exp.physical)
    return dict(feasible=True, Q=Q, paths=paths, alpha=alpha, beta=beta,
                face_drift=abs(face_lhs - Q),
                face_budget_nominal=nominal_face_budget(len(inst.od)))
