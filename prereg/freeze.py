"""Freeze the pre-registration: refuse if any TO_FREEZE remains, else write SHA-256 manifest.

    python prereg/freeze.py            # validate + write prereg/MANIFEST.sha256
    python prereg/freeze.py --check    # verify files still match the manifest (harness calls this)
"""
import hashlib, sys
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["g0_spec.yaml", "c_g0_spec.yaml", "noise_config.json"]
MANIFEST = HERE / "MANIFEST.sha256"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if "--check" in sys.argv:
        if not MANIFEST.exists():
            sys.exit("not frozen: MANIFEST.sha256 missing")
        want = dict(line.split("  ")[::-1] for line in MANIFEST.read_text(encoding="utf-8").split("\n") if line)
        bad = [f for f in FILES if want.get(f) != digest(HERE / f)]
        if bad:
            sys.exit(f"spec changed since freeze: {bad}")
        print("pre-registration verified")
        return
    open_items = {f: (HERE / f).read_text(encoding="utf-8").count("TO_FREEZE") for f in FILES}
    open_items = {f: n for f, n in open_items.items() if n}
    if open_items:
        sys.exit(f"cannot freeze, TO_FREEZE remaining: {open_items}")
    MANIFEST.write_text("".join(f"{digest(HERE / f)}  {f}\n" for f in FILES))
    print(MANIFEST.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
