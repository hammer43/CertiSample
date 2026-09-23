"""Master problem with the validated lexicographic tie-break (spec 12A.1).

min sum_i fbar_i y_i + theta   s.t.  y_i + y_j <= 1 on conflict edges,
                                     theta >= alpha_k + beta_k^T y   (optimality cuts),
                                     gamma_l + delta_l^T y <= 0      (feasibility cuts).
Ties on the primary objective are broken by minimizing y in ascending site order, using
HiGHS native lexicographic multi-objective optimization (preflight PASS_NATIVE, v1.4.1).
"""
from __future__ import annotations

import numpy as np

try:
    import highspy
except ImportError:  # pragma: no cover
    highspy = None

BIG = 1e9


def _linobj(coeffs, priority, abs_tol=0.0, rel_tol=0.0):
    o = highspy.HighsLinearObjective()
    o.weight = 1.0
    o.offset = 0.0
    o.coefficients = np.asarray(coeffs, dtype=np.float64)
    o.abs_tolerance = float(abs_tol)
    o.rel_tolerance = float(rel_tol)
    o.priority = int(priority)
    return o


def solve_master(sites, fbar, conflict_edges, opt_cuts=(), feas_cuts=(),
                 use_theta=True, lexicographic=True, mip_feasibility_tolerance=1e-6):
    """Returns (y dict, objective, theta). opt_cuts: (alpha, beta dict);
    feas_cuts: (gamma, delta dict)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("random_seed", 0)
    h.setOptionValue("mip_feasibility_tolerance", mip_feasibility_tolerance)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.setOptionValue("blend_multi_objectives", False)

    y = {i: h.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger) for i in sites}
    theta = h.addVariable(lb=-BIG, ub=BIG) if use_theta else None
    for i, j in conflict_edges:
        h.addConstr(y[i] + y[j] <= 1)
    for alpha, beta in opt_cuts:
        h.addConstr(theta >= alpha + sum(beta.get(i, 0.0) * y[i] for i in sites))
    for gamma, delta in feas_cuts:
        h.addConstr(gamma + sum(delta.get(i, 0.0) * y[i] for i in sites) <= 0)

    n = len(sites) + (1 if use_theta else 0)
    primary = [fbar[i] for i in sites] + ([1.0] if use_theta else [])
    if lexicographic:
        h.addLinearObjective(_linobj(primary, priority=len(sites) + 10))
        for k, i in enumerate(sites):
            c = [0.0] * n
            c[k] = 1.0
            h.addLinearObjective(_linobj(c, priority=len(sites) - k))
        h.run()
    else:
        h.minimize(sum(fbar[i] * y[i] for i in sites) + (theta if use_theta else 0))
    status = h.modelStatusToString(h.getModelStatus())
    if "Optimal" not in status:
        return None, None, None, status
    yv = {i: int(round(float(h.val(y[i])))) for i in sites}
    th = float(h.val(theta)) if use_theta else 0.0
    obj = sum(fbar[i] * yv[i] for i in sites) + th
    return yv, obj, th, status


def spacing_diagnostics(sites, fbar, conflict_edges, solve_cost):
    """V_space and Delta_space (spec 11A): compare the unconstrained optimum with the
    spacing-constrained one. solve_cost(y) returns the true cost of a candidate."""
    y0, _, _, s0 = solve_master(sites, fbar, [], use_theta=False)
    yr, _, _, s1 = solve_master(sites, fbar, conflict_edges, use_theta=False)
    if y0 is None or yr is None:
        return dict(status=(s0, s1))
    V = sum(1 for i, j in conflict_edges if y0[i] and y0[j])
    z0, zr = solve_cost(y0), solve_cost(yr)
    D = (zr - z0) / max(abs(z0), 1e-9) if z0 is not None and zr is not None else None
    return dict(V_space=V, Delta_space=D, z_unconstrained=z0, z_constrained=zr,
                n_open_unconstrained=sum(y0.values()), n_open_constrained=sum(yr.values()))
