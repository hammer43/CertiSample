"""Load the pre-registered specs and enforce the freeze."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "prereg"


def verify_frozen() -> None:
    r = subprocess.run([sys.executable, str(PREREG / "freeze.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"pre-registration not frozen or modified: {r.stdout}{r.stderr}".strip())


def load(name: str) -> dict:
    p = PREREG / name
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8"))
