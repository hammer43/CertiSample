"""python -m certisample.cli {env, landscape}"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path

import pandas as pd

from .instances import make_instance, seed_range
from .landscape import coverage_radius, is_maximal, omega_eps, swap_components
from .spec import load, verify_frozen


def cmd_env(_):
    """Print installed versions for noise_config.json 'versions'."""
    out = {"python": platform.python_version()}
    for mod in ["qoolqit", "pulser", "pulser_simulation", "numpy", "scipy", "networkx", "ortools"]:
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"MISSING ({type(e).__name__})"
    print(json.dumps(out, indent=2))


def cmd_landscape(a):
    """Stage 0 of A-G0: exact landscapes for pilot (and optionally test) instances.
    Uses only the generator and epsilon rule; no sampler output is involved."""
    if not a.allow_unfrozen:
        verify_frozen()
    spec = load("g0_spec.yaml")
    grid = spec["epsilon_rule"]["grid_relative"]
    min_size = spec["epsilon_rule"]["min_size"]
    gen = spec["instances"]["generator"]
    rows = []
    for split in a.splits.split(","):
        for s in seed_range(spec, split):
            inst = make_instance(s, split, side=gen["side"], occupied=gen["occupied_sites"],
                                 low=gen["weights"]["low"], high=gen["weights"]["high"])
            eps, E_star, omega, flagged = omega_eps(inst.G, inst.w, grid, min_size=min_size)
            comps = swap_components(omega)
            f_max = sum(is_maximal(m, inst.G) for m in omega) / len(omega)
            sizes = [bin(m).count("1") for m in omega]
            rows.append(dict(split=split, seed=s, n_edges=inst.G.number_of_edges(), E_star=E_star,
                             eps_rel=eps, omega_size=len(omega), flagged=flagged,
                             n_components=len(comps), largest_component=max(len(c) for c in comps),
                             f_max=round(f_max, 4), card_min=min(sizes), card_max=max(sizes),
                             radius_r=round(coverage_radius(omega), 4)))
            print(f"  {split} seed {s}: eps={eps} |Omega|={len(omega)} comps={len(comps)} "
                  f"f_max={f_max:.3f} r={rows[-1]['radius_r']}",
                  flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "landscape.csv", index=False)
    print(df.describe().round(3).to_string())


def cmd_a_g0(a):
    from .a_g0 import tune_classical
    sel = tune_classical(Path(a.out), reps=a.reps)
    print(json.dumps({k: sel[k] for k in ("chosen", "comparator", "comparator_table")},
                     indent=2, default=str))


def cmd_rydberg_validate(a):
    from .a_g0_rydberg import validate
    r = validate(Path(a.out))
    print(json.dumps({k: r[k] for k in ("threshold", "max_TV", "passed")}, indent=2))


def cmd_rydberg_tune(a):
    from .a_g0_rydberg import tune
    s = tune(Path(a.out), workers=a.workers)
    print(json.dumps({k: s[k] for k in ("chosen", "infeasible_fraction", "out_of_window_fraction")},
                     indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="certisample")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    L = sub.add_parser("landscape")
    L.add_argument("--splits", default="pilot")
    L.add_argument("--out", default="results/landscape")
    L.add_argument("--allow-unfrozen", action="store_true",
                   help="setup check only; the gates themselves always require a frozen spec")
    A = sub.add_parser("a-g0-tune", help="tune classical arms on pilots and select comparator")
    A.add_argument("--out", default="results/a_g0")
    A.add_argument("--reps", type=int, default=None,
                   help="override replicates (smoke tests only; the frozen value is used by default)")
    V = sub.add_parser("rydberg-validate", help="validation gate: subspace emulator vs exact Pulser")
    V.add_argument("--out", default="results/a_g0")
    R = sub.add_parser("rydberg-tune", help="tune the Rydberg schedule on pilots (resumable)")
    R.add_argument("--out", default="results/a_g0")
    R.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    {"env": cmd_env, "landscape": cmd_landscape, "a-g0-tune": cmd_a_g0,
     "rydberg-validate": cmd_rydberg_validate, "rydberg-tune": cmd_rydberg_tune}[a.cmd](a)


if __name__ == "__main__":
    main()
