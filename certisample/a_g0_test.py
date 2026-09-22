"""A-G0 test stage: the only step that touches the 30 held-out test graphs.

Preconditions (all enforced): frozen spec, classical_selection.json, a passed
rydberg_validation.json, and rydberg_selection.json. Nothing here is tuned.

Pass rule (g0_spec.yaml statistics): average replicates within instance, then
  delta_d = MNJ(comparator) - MNJ(rydberg)      (positive = Rydberg better)
  delta_c = RCF(rydberg) - RCF(comparator)      (positive = Rydberg better)
PASS iff LB95(delta_d) > 0 AND LB95(delta_c) > 0 (paired bootstrap over instances).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .a_g0 import ARM_IDS, _landscape
from .a_g0_rydberg import RYDBERG_ARM_ID, _grid
from .instances import make_instance, seed_range
from .rydberg import Schedule, Subspace
from .samplers import ARMS, Problem
from .scoring import paired_bootstrap_lb
from .spec import PREREG, load, verify_frozen

METRICS = ["mean_nearest_jaccard", "radius_coverage_fraction", "worst_case_jaccard_radius",
           "swap_component_coverage", "unique_in_window", "infeasible_fraction",
           "out_of_window_fraction", "duplicate_fraction"]


def verdict(delta_d, delta_c, resamples: int, seed: int, interval: float) -> dict:
    lb_d = paired_bootstrap_lb(delta_d, resamples, seed, interval)
    lb_c = paired_bootstrap_lb(delta_c, resamples, seed, interval)
    return dict(mean_delta_d=float(np.mean(delta_d)), LB95_delta_d=lb_d,
                mean_delta_c=float(np.mean(delta_c)), LB95_delta_c=lb_c,
                decision="PASS" if (lb_d > 0 and lb_c > 0) else "FAIL")


def _simulate_test(args):
    seed, d, f, cache = args
    path = Path(cache) / f"test{seed}_d{d}_f{f}.npz"
    if path.exists():
        return str(path), 0.0
    inst = make_instance(seed, "test")
    t = time.time()
    S = Subspace(inst.positions, inst.w)
    np.savez_compressed(path, states=S.states, probs=S.evolve(Schedule(d, f)))
    return str(path), time.time() - t


def _feasible(p):
    return lambda m: all(not (p.nb[v] & m) for v in range(p.n) if (m >> v) & 1)


def run_test(out: Path, workers: int = 1, curve: bool = True, limit: int | None = None) -> dict:
    verify_frozen()
    spec = load("g0_spec.yaml")
    need = ["classical_selection.json", "rydberg_validation.json", "rydberg_selection.json"]
    missing = [f for f in need if not (out / f).exists()]
    if missing:
        raise SystemExit(f"missing precondition files in {out}: {missing}")
    if not json.loads((out / "rydberg_validation.json").read_text())["passed"]:
        raise SystemExit("Rydberg validation gate did not pass; the test stage cannot run")
    csel = json.loads((out / "classical_selection.json").read_text())
    rsel = json.loads((out / "rydberg_selection.json").read_text())["chosen"]

    K = spec["sampling"]["K_primary"]
    reps = spec["sampling"]["replicates_per_instance"]
    rep_seeds = list(range(spec["sampling"]["replicate_seeds"][0],
                           spec["sampling"]["replicate_seeds"][0] + reps))
    tests = seed_range(spec, "test")
    official = limit is None
    if limit:
        tests = tests[:limit]
    grid = _grid(spec)
    rci = rsel["config_index"]
    sch = grid[rci]
    assert (sch.duration_units, sch.delta_final_units) == (rsel["duration_units"], rsel["delta_final_units"])

    # ---- Rydberg simulations (one per test graph), cached and resumable
    cache = out / "rydberg_cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(s, sch.duration_units, sch.delta_final_units, str(cache)) for s in tests]
    todo = [j for j in jobs if not (cache / f"test{j[0]}_d{j[1]}_f{j[2]}.npz").exists()]
    print(f"  Rydberg: {len(jobs)} test simulations, {len(jobs)-len(todo)} cached, "
          f"{len(todo)} to run with {workers} worker(s)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, secs) in enumerate(ex.map(_simulate_test, todo), 1):
            print(f"  [{i}/{len(todo)}] {Path(path).name} {secs:.0f}s  elapsed {time.time()-t0:.0f}s",
                  flush=True)

    # ---- scoring: every tuned classical arm (comparator + descriptive) and Rydberg
    Ks = spec["sampling"]["K_secondary_curve"] if curve else [K]
    Ks = sorted(set(Ks) | {K})
    rows = []
    for s in tests:
        inst, scorer = _landscape(spec, s, "test")
        p = Problem(inst.G, inst.w)
        feas = _feasible(p)
        z = np.load(cache / f"test{s}_d{sch.duration_units}_f{sch.delta_final_units}.npz")
        states, probs = z["states"], z["probs"]
        for k in Ks:
            for r in rep_seeds:
                rng = np.random.default_rng([s, r, RYDBERG_ARM_ID, rci])
                outs = [int(states[j]) for j in rng.choice(len(probs), size=k, p=probs)]
                rows.append(dict(arm="rydberg_ideal", K=k, seed=s, rep=r, energy_evals=np.nan,
                                 **scorer.score(outs, feas)))
                for arm, c in csel["chosen"].items():
                    if k != K and arm != csel["comparator"]:
                        continue            # curve only for comparator vs Rydberg
                    rng = np.random.default_rng([s, r, ARM_IDS[arm], c["config_index"]])
                    outs, ev = ARMS[arm](p, k, rng, **c["params"])
                    rows.append(dict(arm=arm, K=k, seed=s, rep=r, energy_evals=ev,
                                     **scorer.score(outs, feas)))
        print(f"  scored test graph {s}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_raw.csv", index=False)

    prim = df[df.K == K]
    inst_mean = prim.groupby(["arm", "seed"])[METRICS].mean()
    comp = csel["comparator"]
    ryd = inst_mean.loc["rydberg_ideal"]
    cmp_ = inst_mean.loc[comp]
    delta_d = (cmp_["mean_nearest_jaccard"] - ryd["mean_nearest_jaccard"]).to_numpy()
    delta_c = (ryd["radius_coverage_fraction"] - cmp_["radius_coverage_fraction"]).to_numpy()
    bs = spec["statistics"]["bootstrap"]
    v = verdict(delta_d, delta_c, bs["resamples"], bs["seed"], bs["interval"])

    per_arm = prim.groupby("arm")[METRICS + ["energy_evals"]].mean().round(4).to_dict("index")
    curve_tbl = (df[df.arm.isin(["rydberg_ideal", comp])]
                 .groupby(["arm", "K"])[["mean_nearest_jaccard", "radius_coverage_fraction"]]
                 .mean().round(4).reset_index().to_dict("records")) if curve else None
    result = dict(
        gate="A-G0", official=official, n_test_instances=len(tests), K=K, replicates=reps,
        comparator=comp, rydberg_schedule=rsel, **v,
        interpretation=spec["claim_if_pass"] if v["decision"] == "PASS" else spec["on_fail"],
        per_arm_means=per_arm, secondary_K_curve=curve_tbl,
        spec_manifest=(PREREG / "MANIFEST.sha256").read_text(encoding="utf-8"))
    name = "a_g0_verdict.json" if official else "a_g0_verdict_DEV_NOT_OFFICIAL.json"
    (out / name).write_text(json.dumps(result, indent=2, default=str))
    return result
