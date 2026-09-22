"""A-G0 classical stage: tune each classical arm on pilots, then select the comparator.

Test-set evaluation is intentionally NOT implemented here: it runs only after the
Rydberg schedule has also been selected on pilots, so no test result can influence tuning.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .instances import make_instance, seed_range
from .landscape import coverage_radius, omega_eps, swap_components
from .samplers import ARMS, IMPLEMENTATION_NOTES, Problem
from .scoring import Scorer
from .spec import PREREG, load, verify_frozen

ARM_IDS = {name: i for i, name in enumerate(sorted(ARMS))}


def _configs(arm_spec: dict):
    grid = arm_spec["grid"]
    fixed = {k: v for k, v in arm_spec.items() if k.endswith("_sweeps") and not isinstance(v, list)}
    fixed.update({k: v for k, v in grid.items() if not isinstance(v, list)})
    varying = {k: v for k, v in grid.items() if isinstance(v, list)}
    keys = sorted(varying)
    for combo in itertools.product(*(varying[k] for k in keys)):
        yield dict(zip(keys, combo), **fixed)


def _landscape(spec, seed, split):
    gen = spec["instances"]["generator"]
    inst = make_instance(seed, split, side=gen["side"], occupied=gen["occupied_sites"],
                         low=gen["weights"]["low"], high=gen["weights"]["high"])
    eps, E_star, omega, flagged = omega_eps(inst.G, inst.w, spec["epsilon_rule"]["grid_relative"],
                                            min_size=spec["epsilon_rule"]["min_size"])
    scorer = Scorer(omega, coverage_radius(omega), swap_components(omega))
    return inst, scorer


def tune_classical(out: Path, reps: int | None = None, verbose=True) -> dict:
    verify_frozen()
    spec = load("g0_spec.yaml")
    K = spec["sampling"]["K_primary"]
    reps = reps or spec["sampling"]["replicates_per_instance"]
    rep_seeds = list(range(spec["sampling"]["replicate_seeds"][0],
                           spec["sampling"]["replicate_seeds"][0] + reps))
    pilots = seed_range(spec, "pilot")
    arms = spec["arms"]["classical"]
    out.mkdir(parents=True, exist_ok=True)

    lands = {s: _landscape(spec, s, "pilot") for s in pilots}
    rows = []
    for arm, aspec in arms.items():
        fn = ARMS[arm]
        for ci, cfg in enumerate(_configs(aspec)):
            t0 = time.time()
            for s in pilots:
                inst, scorer = lands[s]
                p = Problem(inst.G, inst.w)
                feas = lambda m, p=p: all(not (p.nb[v] & m) for v in range(p.n) if (m >> v) & 1)
                for r in rep_seeds:
                    rng = np.random.default_rng([s, r, ARM_IDS[arm], ci])
                    outs, evals = fn(p, K, rng, **cfg)
                    rows.append(dict(arm=arm, config=ci, **{f"cfg_{k}": v for k, v in cfg.items()},
                                     seed=s, rep=r, energy_evals=evals, **scorer.score(outs, feas)))
            if verbose:
                print(f"  {arm} config {ci} {cfg}: {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pilot_tuning_raw.csv", index=False)

    # average replicates within instance first (frozen aggregation rule)
    inst_mean = (df.groupby(["arm", "config", "seed"])
                   [["mean_nearest_jaccard", "radius_coverage_fraction"]].mean().reset_index())

    def mean_rank(frame, key):
        """rank every candidate on every pilot instance on both co-primaries; mean rank."""
        r_a = frame.groupby("seed")["mean_nearest_jaccard"].rank(method="average", ascending=True)
        r_b = frame.groupby("seed")["radius_coverage_fraction"].rank(method="average", ascending=False)
        frame = frame.assign(rank=(r_a + r_b) / 2)
        return (frame.groupby(key)
                     .agg(mean_rank=("rank", "mean"), mnj=("mean_nearest_jaccard", "mean"),
                          rcf=("radius_coverage_fraction", "mean"))
                     .sort_values(["mean_rank", "mnj"]))

    chosen = {}
    for arm in arms:
        sub = inst_mean[inst_mean.arm == arm]
        table = mean_rank(sub, "config")
        best = int(table.index[0])
        cfg = next(c for i, c in enumerate(_configs(arms[arm])) if i == best)
        chosen[arm] = dict(config_index=best, params=cfg,
                           pilot_mean_rank=float(table.iloc[0].mean_rank),
                           pilot_mean_nearest_jaccard=float(table.iloc[0].mnj),
                           pilot_radius_coverage=float(table.iloc[0].rcf))
    tuned = pd.concat([inst_mean[(inst_mean.arm == a) & (inst_mean.config == c["config_index"])]
                       for a, c in chosen.items()])
    comp_table = mean_rank(tuned, "arm")
    comparator = str(comp_table.index[0])

    manifest = (PREREG / "MANIFEST.sha256").read_text(encoding="utf-8")
    selection = dict(stage="A-G0 classical pilot tuning", spec_manifest=manifest,
                     K=K, replicates=reps, pilots=pilots, chosen=chosen,
                     comparator=comparator,
                     comparator_table=comp_table.reset_index().to_dict("records"),
                     implementation_notes=IMPLEMENTATION_NOTES,
                     note="test set not touched; runs only after Rydberg schedule selection")
    (out / "classical_selection.json").write_text(json.dumps(selection, indent=2, default=str))
    return selection
