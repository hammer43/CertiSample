"""Nearest-neighbour blockade-subspace emulator for the A-G0 Rydberg arm (spec v3).

Hamiltonian (Pulser convention, ground-rydberg basis, hbar = 1, rad/us and us):
    H(t) = sum_i Omega(t)/2 X_i - sum_i (delta(t) + eps_i * Delta_dmm) n_i
           + sum_{i<j} C6 / r_ij^6 n_i n_j
The state space keeps configurations with at most `max_nn_pairs` (default 1) pairs of
excited atoms at lattice distance 1 (nearest neighbours, J/Omega ~ 24). Diagonal and longer-range pairs stay in
the dynamics. States beyond the kept level are eliminated adiabatically to second order:
each kept state receives the energy shift -(Omega/2)^2 / (E_excluded - E_allowed) summed
over its excluded single-flip neighbours. The restriction is an approximation, validated against the exact Pulser
QutipEmulator by `certisample.cli rydberg-validate` before any Rydberg data is drawn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

C6 = 5420158.53          # rad um^6 / us, pulser DigitalAnalogDevice (n = 70)
SPACING_UM = 5.0
OMEGA_TILDE = 1.7 ** -6  # Omega_max / J_nn, so that the blockade radius is 1.7 a
J_NN = C6 / SPACING_UM ** 6
OMEGA_MAX = OMEGA_TILDE * J_NN   # ~14.37 rad/us
RAMP_FRACTION = 0.1


@dataclass(frozen=True)
class Schedule:
    duration_units: float     # T = duration_units / OMEGA_MAX
    delta_final_units: float  # delta_final = delta_final_units * OMEGA_MAX

    @property
    def T_us(self) -> float:
        ns = 4 * round(self.duration_units / OMEGA_MAX * 1000 / 4)   # device clock: 4 ns
        return ns / 1000.0

    @property
    def delta_final(self) -> float:
        return self.delta_final_units * OMEGA_MAX

    @property
    def T_ns(self) -> int:
        return int(round(self.T_us * 1000))

    @property
    def ramp_ns(self) -> int:
        return 4 * int(round(RAMP_FRACTION * self.T_ns / 4))

    def samples(self):
        """Per-nanosecond (Omega, delta) samples, identical to Pulser's waveform sampling:
        RampWaveform(d, a, b) = linspace(a, b, d); ConstantWaveform(d, a) = a."""
        T, r = self.T_ns, self.ramp_ns
        omega = np.concatenate([np.linspace(0.0, OMEGA_MAX, r),
                                np.full(T - 2 * r, OMEGA_MAX),
                                np.linspace(OMEGA_MAX, 0.0, r)])
        delta = np.linspace(-self.delta_final, self.delta_final, T)
        return omega, delta


class Subspace:
    """Basis of blockade-respecting (distance-1) configurations for one register."""

    def __init__(self, positions: list, weights: dict, max_nn_pairs: int = 1):
        self.n = n = len(positions)
        self.pos = [(x * SPACING_UM, y * SPACING_UM) for x, y in positions]
        nn = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1]) == 1:
                    nn[i] |= 1 << j
                    nn[j] |= 1 << i
        states = []

        def rec(i, m, pairs):
            if i == n:
                states.append(m)
                return
            rec(i + 1, m, pairs)
            add = bin(nn[i] & m).count("1")          # new nearest-neighbour pairs
            if pairs + add <= max_nn_pairs:
                rec(i + 1, m | (1 << i), pairs + add)
        rec(0, 0, 0)
        self.max_nn_pairs = max_nn_pairs
        self.states = np.array(states, dtype=np.int64)
        index = {int(s): k for k, s in enumerate(states)}
        D = len(states)
        occ = np.array([[(s >> i) & 1 for i in range(n)] for s in states], dtype=float)
        # interaction energy (all pairs, 1/r^6)
        Jij = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = math.dist(self.pos[i], self.pos[j])
                Jij[i, j] = C6 / r ** 6
        self.E_int = np.einsum("ki,ij,kj->k", occ, Jij, occ)
        self.N = occ.sum(1)
        self.eps = np.array([1.0 - weights[i] for i in range(n)])
        self.E_eps = occ @ self.eps
        rows, cols = [], []
        for k, s in enumerate(states):
            for i in range(n):
                t = s ^ (1 << i)
                if t in index:
                    rows.append(k)
                    cols.append(index[t])
        self.X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(D, D))
        self.ground = index[0]
        # excluded single flips (exciting atom i creates a nearest-neighbour pair):
        # kept only through second-order adiabatic elimination (energy shift below)
        ex_s, ex_gap_int, ex_eps, ex_atom = [], [], [], []
        for k, s_ in enumerate(states):
            for i in range(n):
                if (s_ >> i) & 1 or (s_ | (1 << i)) in index:
                    continue
                dE = sum(Jij[min(i, j), max(i, j)] for j in range(n) if (s_ >> j) & 1)
                ex_s.append(k)
                ex_gap_int.append(dE)
                ex_eps.append(self.eps[i])
                ex_atom.append(i)
        self.ex_s = np.array(ex_s, dtype=np.int64)
        self.ex_gap_int = np.array(ex_gap_int)
        self.ex_eps = np.array(ex_eps)
        self._ex_atom = np.array(ex_atom, dtype=np.int64)

    def set_fields(self, weights: dict) -> None:
        """Change only the local fields (DMM weights); the register is untouched."""
        self.eps = np.array([1.0 - weights[i] for i in range(self.n)])
        occ = np.array([[(int(s) >> i) & 1 for i in range(self.n)] for s in self.states], dtype=float)
        self.E_eps = occ @ self.eps
        if len(self.ex_s):
            self.ex_eps = self.eps[self._ex_atom]

    def evolve_arrays(self, omega, delta, dmm_amp: float) -> np.ndarray:
        """Final-state probabilities for per-nanosecond (Omega, delta) samples and a constant
        DMM amplitude dmm_amp >= 0 (applied as -dmm_amp * eps_i). Initial state: all ground."""
        from scipy.sparse import diags
        from scipy.sparse.linalg import expm_multiply
        static = self.E_int + dmm_amp * self.E_eps
        psi = np.zeros(len(self.states), complex)
        psi[self.ground] = 1.0
        D = len(self.states)
        for om, de in zip(omega, delta):
            diag = static - de * self.N
            if len(self.ex_s) and om > 0:
                gap = self.ex_gap_int + dmm_amp * self.ex_eps - de
                diag = diag - (0.5 * om) ** 2 * np.bincount(self.ex_s, 1.0 / gap, minlength=D)
            H = diags(diag) + (0.5 * om) * self.X
            psi = expm_multiply(-1j * 1e-3 * H, psi)
        p = np.abs(psi) ** 2
        return p / p.sum()

    def evolve(self, sch: Schedule) -> np.ndarray:
        """A-G0 schedule family (constant DMM amplitude = delta_final)."""
        omega, delta = sch.samples()
        return self.evolve_arrays(omega, delta, sch.delta_final)

    def sample(self, probs: np.ndarray, K: int, rng) -> list[int]:
        return [int(self.states[k]) for k in rng.choice(len(probs), size=K, p=probs)]


def pulser_exact_probs(positions, weights, sch: Schedule, **qutip_options) -> dict:
    """Exact reference via pulser_simulation.QutipEmulator (full Hilbert space)."""
    from pulser import Pulse, Register, Sequence
    from pulser.devices import DigitalAnalogDevice
    from pulser.waveforms import CompositeWaveform, ConstantWaveform, RampWaveform
    n = len(positions)
    reg = Register({f"q{i}": (x * SPACING_UM, y * SPACING_UM) for i, (x, y) in enumerate(positions)})
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("r", "rydberg_global")
    dmap = reg.define_detuning_map({f"q{i}": 1.0 - weights[i] for i in range(n)})
    seq.config_detuning_map(dmap, "dmm_0")
    T, r = sch.T_ns, sch.ramp_ns
    amp = CompositeWaveform(RampWaveform(r, 0.0, OMEGA_MAX),
                            ConstantWaveform(T - 2 * r, OMEGA_MAX),
                            RampWaveform(r, OMEGA_MAX, 0.0))
    seq.add_dmm_detuning(ConstantWaveform(T, -sch.delta_final), "dmm_0")
    seq.add(Pulse(amp, RampWaveform(T, -sch.delta_final, sch.delta_final), 0), "r",
            protocol="no-delay")
    from pulser_simulation import QutipEmulator
    res = QutipEmulator.from_sequence(seq).run(**qutip_options)
    psi = res.get_final_state().full().ravel()
    probs = np.abs(psi) ** 2
    out = {}
    for idx in np.nonzero(probs > 1e-12)[0]:
        bits = format(int(idx), f"0{n}b")            # q0 is the most significant digit
        m = sum(1 << i for i, b in enumerate(bits) if b == "0")   # basis 0 = 'r' = excited
        out[m] = out.get(m, 0.0) + float(probs[idx])
    return out


def total_variation(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def pulser_exact_probs_arrays(positions, weights, omega, delta, dmm_amp: float, **qutip_options) -> dict:
    """Exact Pulser reference for arbitrary per-ns samples (CustomWaveform)."""
    from pulser import Pulse, Register, Sequence
    from pulser.devices import DigitalAnalogDevice
    from pulser.waveforms import ConstantWaveform, CustomWaveform
    from pulser_simulation import QutipEmulator
    n = len(positions)
    reg = Register({f"q{i}": (x * SPACING_UM, y * SPACING_UM) for i, (x, y) in enumerate(positions)})
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("r", "rydberg_global")
    seq.config_detuning_map(reg.define_detuning_map({f"q{i}": 1.0 - weights[i] for i in range(n)}), "dmm_0")
    T = len(omega)
    seq.add_dmm_detuning(ConstantWaveform(T, -dmm_amp), "dmm_0")
    seq.add(Pulse(CustomWaveform(np.asarray(omega)), CustomWaveform(np.asarray(delta)), 0), "r",
            protocol="no-delay")
    psi = QutipEmulator.from_sequence(seq).run(**qutip_options).get_final_state().full().ravel()
    probs = np.abs(psi) ** 2
    out = {}
    for idx in np.nonzero(probs > 1e-12)[0]:
        bits = format(int(idx), f"0{n}b")
        m = sum(1 << i for i, b in enumerate(bits) if b == "0")
        out[m] = out.get(m, 0.0) + float(probs[idx])
    return out
