"""C-G0: fixed-register variational continuation (prereg/c_g0_spec.yaml v3).

Stages, in order (each refuses to run without its predecessor):
  delta0   : classical Lagrangian surrogate on the 20 pilot masters -> frozen delta0
  validate : subspace emulator vs tight exact Pulser on the C-G0 schedule family
  test     : per (test graph, replicate): cold SPSA on h0 -> theta*(h0); then on h1, warm from
             theta*(h0) vs cold from the canonical schedule. Resumable, parallel.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .instances import make_instance, seed_range
from .landscape import all_independent_sets
from .rydberg import (OMEGA_MAX, Subspace, pulser_exact_probs_arrays, total_variation)
from .scoring import paired_bootstrap_lb
from .spec import PREREG, load, verify_frozen

TV_THRESHOLD = 0.005
REFERENCE_OPTIONS = dict(atol=1e-12, rtol=1e-10, nsteps=10 ** 7)
DMM_AMP_UNITS = 4.0
COLD = np.array([1.0, 1.0, 1.0, 1.0, -2.4, -0.8, 0.8, 2.4])
LO = np.array([0.0] * 4 + [-6.0] * 4)
HI = np.array([1.0] * 4 + [6.0] * 4)


# ---------------------------------------------------------------- problem pieces
def _spec():
    return load("c_g0_spec.yaml")


def instance(seed: int, split: str):
    g = _spec()["instances"]["generator"]
    return make_instance(seed, split, side=g["side"], occupied=g["occupied_sites"],
                         low=g["weights"]["low"], high=g["weights"]["high"])


def T_ns() -> int:
    return 4 * round(_spec()["ansatz"]["duration"] / OMEGA_MAX * 1000 / 4)


def samples(theta: np.ndarray, T: int):
    th = np.clip(theta, LO, HI)
    knots = np.linspace(0, T - 1, 6)
    om = np.r_[0.0, th[:4], 0.0] * OMEGA_MAX
    de = np.r_[-4.0, th[4:], 4.0] * OMEGA_MAX
    t = np.arange(T)
    return np.interp(t, knots, om), np.interp(t, knots, de)


def mwis(G, h: dict):
    best, arg = -1.0, 0
    for m in all_independent_sets(G):
        v = sum(h[i] for i in h if (m >> i) & 1)
        if v > best:
            best, arg = v, m
    return arg, best


# ---------------------------------------------------------------- delta0 (pilots, classical)
def lagrangian_deltas(seed: int, rounds: int = 30, n_cons: int = 3, frac: float = 0.6) -> list[float]:
    """Budget-constraint Lagrangian surrogate (spec v4); constraints bind by construction."""
    inst = instance(seed, "pilot")
    n = len(inst.positions)
    rng = np.random.default_rng(seed)
    w = np.array([inst.w[i] for i in range(n)])
    sets = all_independent_sets(inst.G)
    occ = np.array([[(m >> i) & 1 for i in range(n)] for m in sets], dtype=float)
    x0 = occ[int(np.argmax(occ @ w))]
    C, B = np.zeros((n_cons, n)), np.zeros(n_cons)
    for j in range(n_cons):
        half = rng.choice(n, size=n // 2, replace=False)
        C[j, half] = rng.uniform(0.5, 1.0, size=n // 2)
        B[j] = frac * (C[j] @ x0)
    mu, hs = np.zeros(n_cons), []
    for t in range(rounds + 1):
        h = w - C.T @ mu
        hs.append(h)
        x = occ[int(np.argmax(occ @ h))]
        mu = np.maximum(0.0, mu + (1.0 / (t + 1)) * (C @ x - B))
    return [float(np.linalg.norm(hs[t + 1] - hs[t]) / max(np.linalg.norm(hs[t]), 1e-9))
            for t in range(rounds)]


def stage_delta0(out: Path) -> dict:
    verify_frozen()
    spec = _spec()
    deltas = []
    for s in seed_range(spec, "pilot"):
        deltas += lagrangian_deltas(s)
    d0 = float(np.median(deltas))
    res = dict(stage="C-G0 delta0", delta0=d0, n_updates=len(deltas),
               quartiles=[float(q) for q in np.quantile(deltas, [0.25, 0.5, 0.75])],
               spec_manifest=(PREREG / "MANIFEST.sha256").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "c_g0_delta0.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------- validation
def stage_validate(out: Path) -> dict:
    verify_frozen()
    T = T_ns()
    rng = np.random.default_rng(0)
    thetas = [("cold", COLD)] + [(f"random{k}", rng.uniform(LO, HI)) for k in range(3)]
    rows = []
    for s in (2000, 2001):
        inst = instance(s, "pilot")
        S = Subspace(inst.positions, inst.w)
        for name, th in thetas:
            om, de = samples(th, T)
            p = S.evolve_arrays(om, de, DMM_AMP_UNITS * OMEGA_MAX)
            sub = {int(S.states[k]): float(p[k]) for k in range(len(p)) if p[k] > 1e-12}
            ex = pulser_exact_probs_arrays(inst.positions, inst.w, om, de,
                                           DMM_AMP_UNITS * OMEGA_MAX, **REFERENCE_OPTIONS)
            tv = total_variation(sub, ex)
            rows.append(dict(seed=s, schedule=name, TV=tv))
            print(f"  seed {s} {name}: TV={tv:.5f}", flush=True)
    worst = max(r["TV"] for r in rows)
    res = dict(stage="C-G0 validation", threshold=TV_THRESHOLD, max_TV=worst,
               passed=bool(worst <= TV_THRESHOLD), cases=rows, reference_options=REFERENCE_OPTIONS,
               spec_manifest=(PREREG / "MANIFEST.sha256").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "c_g0_validation.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------- optimization
class Objective:
    """Mean sampled energy; infeasible samples score 0 (spec v4). Target closes rho = 0.25
    of the cold schedule's gap: E* + RHO * (E_cold - E*)."""
    RHO = 0.75

    def __init__(self, S: Subspace, G, h: dict, T: int):
        self.S, self.T = S, T
        n = len(h)
        nb = [0] * n
        for u, v in G.edges():
            nb[u] |= 1 << v
            nb[v] |= 1 << u
        st = [int(x) for x in S.states]
        feas = np.array([all(not (nb[v] & m) for v in range(n) if (m >> v) & 1) for m in st])
        hv = np.array([h[i] for i in range(n)])
        occ = np.array([[(m >> i) & 1 for i in range(n)] for m in st], dtype=float)
        self.E = np.where(feas, -(occ @ hv), 0.0)
        opt, best = mwis(G, h)
        self.E_star = -best
        self.opt_index = st.index(opt)
        self.E_cold = float((self.probs(COLD) * self.E).sum())
        self.target = self.E_star + self.RHO * (self.E_cold - self.E_star)

    def probs(self, theta):
        om, de = samples(theta, self.T)
        return self.S.evolve_arrays(om, de, DMM_AMP_UNITS * OMEGA_MAX)

    def value(self, theta, shots, rng):
        p = self.probs(theta)
        if shots == "exact":
            return float((p * self.E).sum()), p
        return float(self.E[rng.choice(len(p), size=shots, p=p)].mean()), p


def spsa(obj: Objective, theta0, rng, shots, n_max, a=0.2, c=0.1, alpha=0.602, gamma=0.101):
    """Returns (N_eval, censored, final theta, p_succ). Every evaluation counts."""
    theta = np.clip(np.array(theta0, float), LO, HI)
    n_eval = 1
    f, p = obj.value(theta, shots, rng)
    if f <= obj.target:
        return n_eval, False, theta, float(p[obj.opt_index])
    k = 0
    while n_eval + 3 <= n_max:
        ak, ck = a / (k + 1) ** alpha, c / (k + 1) ** gamma
        d = rng.choice([-1.0, 1.0], size=len(theta))
        fp, _ = obj.value(np.clip(theta + ck * d, LO, HI), shots, rng)
        fm, _ = obj.value(np.clip(theta - ck * d, LO, HI), shots, rng)
        theta = np.clip(theta - ak * (fp - fm) / (2 * ck * d), LO, HI)
        f, p = obj.value(theta, shots, rng)
        n_eval += 3
        k += 1
        if f <= obj.target:
            return n_eval, False, theta, float(p[obj.opt_index])
    return n_max, True, theta, float(p[obj.opt_index])


def _job(args):
    seed, rep, delta, shots, n_max, cache = args
    path = Path(cache) / f"test{seed}_rep{rep}_d{delta:.6g}_s{shots}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    inst = instance(seed, "test")
    n = len(inst.positions)
    h0 = {i: inst.w[i] for i in range(n)}
    u = np.random.default_rng([seed, 7]).normal(size=n)
    u /= np.linalg.norm(u)
    v0 = np.array([h0[i] for i in range(n)])
    h1v = np.clip(v0 + delta * np.linalg.norm(v0) * u, 0.0, 1.0)
    h1 = {i: float(h1v[i]) for i in range(n)}
    T = T_ns()
    S = Subspace(inst.positions, h0)
    obj0 = Objective(S, inst.G, h0, T)
    n0, c0, theta_star, _ = spsa(obj0, COLD, np.random.default_rng([seed, rep, 0]), shots, n_max)
    S.set_fields(h1)
    obj1 = Objective(S, inst.G, h1, T)
    nw, cw, _, pw = spsa(obj1, theta_star, np.random.default_rng([seed, rep, 1]), shots, n_max)
    nc, cc, _, pc = spsa(obj1, COLD, np.random.default_rng([seed, rep, 2]), shots, n_max)
    opt0, _ = mwis(inst.G, h0)
    opt1, _ = mwis(inst.G, h1)
    res = dict(seed=seed, rep=rep, delta=delta, shots=shots, N_h0=n0, cens_h0=c0, N_warm=nw, cens_warm=cw, p_warm=pw,
               N_cold=nc, cens_cold=cc, p_cold=pc, optimum_moved=bin(opt0 ^ opt1).count("1"),
               seconds=round(time.time() - t0, 1))
    path.write_text(json.dumps(res))
    return res


def stage_test(out: Path, workers: int = 1, limit=None, reps=None, n_max=None) -> dict:
    verify_frozen()
    spec = _spec()
    for f, key in (("c_g0_delta0.json", None), ("c_g0_validation.json", "passed")):
        if not (out / f).exists():
            raise SystemExit(f"missing {f}: run the earlier C-G0 stage first")
        if key and not json.loads((out / f).read_text())[key]:
            raise SystemExit("C-G0 validation gate did not pass; the test stage cannot run")
    delta0 = json.loads((out / "c_g0_delta0.json").read_text())["delta0"]
    opt = spec["optimization"]
    shots, cap = opt["shots_primary"], (n_max or opt["N_max"])
    R = reps or spec["statistics"]["replicates_per_instance"]
    tests = seed_range(spec, "test")
    official = limit is None and reps is None and n_max is None
    if limit:
        tests = tests[:limit]
    cache = out / "c_g0_cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r, delta0, shots, cap, str(cache)) for s in tests for r in range(R)]
    print(f"  C-G0: {len(jobs)} jobs (graphs x replicates), delta0={delta0:.4g}, "
          f"{workers} worker(s)", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs), 1):
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] graph {r['seed']} rep {r['rep']}: N_warm={r['N_warm']} "
                  f"N_cold={r['N_cold']} ({r['seconds']}s)  elapsed {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "c_g0_raw.csv", index=False)
    g = df.groupby("seed").agg(Nw=("N_warm", "mean"), Nc=("N_cold", "mean"),
                               pw=("p_warm", "mean"), pc=("p_cold", "mean"),
                               moved=("optimum_moved", "first"))
    T_i = (1 - g.Nw / g.Nc).to_numpy()
    dp = (g.pw - g.pc).to_numpy()
    bs = spec["statistics"]["bootstrap"]
    lb_T = paired_bootstrap_lb(T_i, bs["resamples"], bs["seed"], bs["interval"])
    lb_p = paired_bootstrap_lb(dp, bs["resamples"], bs["seed"], bs["interval"])
    decision = "PASS" if (lb_T > 0 and lb_p > -0.02) else "FAIL"
    res = dict(gate="C-G0", official=official, n_test_instances=len(tests), replicates=R,
               delta0=delta0, N_max=cap, mean_T=float(np.mean(T_i)), LB95_T=lb_T,
               mean_delta_p_succ=float(np.mean(dp)), LB95_delta_p_succ=lb_p,
               f_cens_warm=float(df.cens_warm.mean()), f_cens_cold=float(df.cens_cold.mean()),
               f_cens_h0=float(df.cens_h0.mean()),
               note_h0="theta*(h0) comes from a run that may itself be censored; reported, not excluded",
               frac_optimum_moved=float((g.moved > 0).mean()),
               T_when_moved=float(np.mean(T_i[g.moved.to_numpy() > 0])) if (g.moved > 0).any() else None,
               T_when_unmoved=float(np.mean(T_i[g.moved.to_numpy() == 0])) if (g.moved == 0).any() else None,
               decision=decision,
               interpretation=("fixed-register warm starts reach the same quality in fewer evaluations"
                               if decision == "PASS" else spec["on_fail"]),
               spec_manifest=(PREREG / "MANIFEST.sha256").read_text(encoding="utf-8"))
    name = "c_g0_verdict.json" if official else "c_g0_verdict_DEV_NOT_OFFICIAL.json"
    (out / name).write_text(json.dumps(res, indent=2, default=str))
    return res
