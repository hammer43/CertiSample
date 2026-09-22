"""A-G0 Rydberg stage (spec v3): validation gate, then pilot tuning of the schedule.

  validate  : exact Pulser vs subspace emulator on the pre-registered cases; writes
              rydberg_validation.json and refuses tuning unless max TV <= threshold.
  tune      : one simulation per (pilot graph, schedule), cached as .npz so the run is
              resumable; replicates resample K shots with frozen seeds; schedule chosen
              by mean rank across the two co-primaries (tie-break: mean nearest Jaccard).
The test set is not touched here.
"""
from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .a_g0 import _landscape
from .instances import make_instance, seed_range
from .rydberg import Schedule, Subspace, pulser_exact_probs, total_variation
from .samplers import Problem
from .spec import PREREG, load, verify_frozen

TV_THRESHOLD = 0.005
REFERENCE_OPTIONS = dict(atol=1e-12, rtol=1e-10, nsteps=10 ** 7)
RYDBERG_ARM_ID = 99


def _grid(spec):
    g = spec["arms"]["quantum"]["rydberg_ideal"]["grid"]
    return [Schedule(d, f) for d, f in itertools.product(g["duration_units"], g["delta_final_units"])]


# ---------------------------------------------------------------- validation gate
def validate(out: Path) -> dict:
    verify_frozen()
    spec = load("g0_spec.yaml")
    grid = _grid(spec)
    shortest = min(grid, key=lambda s: (s.T_ns, s.delta_final_units))
    longest = max(grid, key=lambda s: (s.T_ns, s.delta_final_units))
    cases = [(12, s) for s in grid] + [(14, shortest), (14, longest)]
    rows = []
    for seed in (0, 1):
        inst = make_instance(seed, "pilot")
        for n, sch in cases:
            pos, w = inst.positions[:n], {i: inst.w[i] for i in range(n)}
            t = time.time()
            S = Subspace(pos, w)
            p = S.evolve(sch)
            sub = {int(S.states[k]): float(p[k]) for k in range(len(p)) if p[k] > 1e-12}
            exact = pulser_exact_probs(pos, w, sch, **REFERENCE_OPTIONS)
            tv = total_variation(sub, exact)
            rows.append(dict(seed=seed, n=n, duration_units=sch.duration_units,
                             delta_final_units=sch.delta_final_units, TV=tv,
                             seconds=round(time.time() - t, 1)))
            print(f"  seed {seed} n={n} ({sch.duration_units},{sch.delta_final_units}): TV={tv:.5f}",
                  flush=True)
    worst = max(r["TV"] for r in rows)
    res = dict(stage="A-G0 Rydberg validation", threshold=TV_THRESHOLD, max_TV=worst,
               reference_options=REFERENCE_OPTIONS,
               passed=bool(worst <= TV_THRESHOLD), cases=rows,
               spec_manifest=(PREREG / "MANIFEST.sha256").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "rydberg_validation.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------- tuning
def _simulate(args):
    seed, d, f, cache = args
    path = Path(cache) / f"pilot{seed}_d{d}_f{f}.npz"
    if path.exists():
        return str(path), 0.0
    inst = make_instance(seed, "pilot")
    t = time.time()
    S = Subspace(inst.positions, inst.w)
    p = S.evolve(Schedule(d, f))
    np.savez_compressed(path, states=S.states, probs=p)
    return str(path), time.time() - t


def tune(out: Path, workers: int = 1) -> dict:
    verify_frozen()
    val = out / "rydberg_validation.json"
    if not val.exists() or not json.loads(val.read_text())["passed"]:
        raise SystemExit("validation gate not passed: run `certisample.cli rydberg-validate` first")
    spec = load("g0_spec.yaml")
    K = spec["sampling"]["K_primary"]
    reps = spec["sampling"]["replicates_per_instance"]
    rep_seeds = list(range(spec["sampling"]["replicate_seeds"][0],
                           spec["sampling"]["replicate_seeds"][0] + reps))
    pilots = seed_range(spec, "pilot")
    grid = _grid(spec)
    cache = out / "rydberg_cache"
    cache.mkdir(parents=True, exist_ok=True)

    jobs = [(s, sch.duration_units, sch.delta_final_units, str(cache)) for s in pilots for sch in grid]
    todo = [j for j in jobs if not (cache / f"pilot{j[0]}_d{j[1]}_f{j[2]}.npz").exists()]
    print(f"  {len(jobs)} simulations, {len(jobs) - len(todo)} cached, {len(todo)} to run "
          f"with {workers} worker(s)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, secs) in enumerate(ex.map(_simulate, todo), 1):
            print(f"  [{i}/{len(todo)}] {Path(path).name} {secs:.0f}s  elapsed {time.time()-t0:.0f}s",
                  flush=True)

    rows = []
    for s in pilots:
        inst, scorer = _landscape(spec, s, "pilot")
        p = Problem(inst.G, inst.w)
        feas = lambda m, p=p: all(not (p.nb[v] & m) for v in range(p.n) if (m >> v) & 1)
        for ci, sch in enumerate(grid):
            z = np.load(cache / f"pilot{s}_d{sch.duration_units}_f{sch.delta_final_units}.npz")
            states, probs = z["states"], z["probs"]
            for r in rep_seeds:
                rng = np.random.default_rng([s, r, RYDBERG_ARM_ID, ci])
                outs = [int(states[k]) for k in rng.choice(len(probs), size=K, p=probs)]
                rows.append(dict(config=ci, duration_units=sch.duration_units,
                                 delta_final_units=sch.delta_final_units, seed=s, rep=r,
                                 **scorer.score(outs, feas)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "rydberg_pilot_raw.csv", index=False)
    inst_mean = (df.groupby(["config", "seed"])
                   [["mean_nearest_jaccard", "radius_coverage_fraction"]].mean().reset_index())
    r_a = inst_mean.groupby("seed")["mean_nearest_jaccard"].rank(ascending=True)
    r_b = inst_mean.groupby("seed")["radius_coverage_fraction"].rank(ascending=False)
    table = (inst_mean.assign(rank=(r_a + r_b) / 2).groupby("config")
             .agg(mean_rank=("rank", "mean"), mnj=("mean_nearest_jaccard", "mean"),
                  rcf=("radius_coverage_fraction", "mean")).sort_values(["mean_rank", "mnj"]))
    best = int(table.index[0])
    sel = dict(stage="A-G0 Rydberg pilot tuning", K=K, replicates=reps, pilots=pilots,
               chosen=dict(config_index=best, duration_units=grid[best].duration_units,
                           delta_final_units=grid[best].delta_final_units,
                           T_ns=grid[best].T_ns,
                           pilot_mean_nearest_jaccard=float(table.iloc[0].mnj),
                           pilot_radius_coverage=float(table.iloc[0].rcf)),
               table=table.reset_index().to_dict("records"),
               infeasible_fraction=float(df.infeasible_fraction.mean()),
               out_of_window_fraction=float(df.out_of_window_fraction.mean()),
               spec_manifest=(PREREG / "MANIFEST.sha256").read_text(encoding="utf-8"),
               note="test set not touched")
    (out / "rydberg_selection.json").write_text(json.dumps(sel, indent=2, default=str))
    return sel
