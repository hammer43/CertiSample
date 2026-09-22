# certisample

Falsification-gated evaluation of quantum-sampled multi-cut Benders on neutral atoms.
Principle: **the quantum device proposes, classical mathematics certifies.**

## Setup (Windows, PowerShell, from C:\CertiSamp)

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest -q
python -m certisample.cli env
```

## Freeze the pre-registration

1. Paste the `env` output into `prereg/noise_config.json` -> `versions`.
2. Fill the noisy-arm values from Pasqal's published device specification and cite it
   in `source_of_values`. No built-in Pulser device provides default noise values.
3. Confirm the scientific choices in `g0_spec.yaml` and `c_g0_spec.yaml`.
4. `python prereg/freeze.py`  (refuses while any TO_FREEZE remains; writes MANIFEST.sha256)
5. Commit the prereg folder and manifest.

## Stage 0 (runs now, before freezing, as a setup check)

```
python -m certisample.cli landscape --splits pilot --allow-unfrozen
```

Computes the exact near-optimal landscape for the 20 pilot graphs: chosen epsilon,
|Omega_eps|, swap components, and the pilot diagnostic f_max. No sampler is involved,
so this is safe to run before freezing. The gates themselves always require a frozen spec.

## Status

| gate | status |
|---|---|
| landscape (stage 0) | implemented |
| A-G0 classical arms | next |
| A-G0 Rydberg arms | after the classical arms |
| C-G0 | after A-G0 |
