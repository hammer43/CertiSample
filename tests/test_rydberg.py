import numpy as np

import certisample.rydberg as R
from certisample.instances import make_instance


def test_schedule_units_and_limits():
    s = R.Schedule(40, 6)
    assert abs(R.OMEGA_MAX - R.J_NN * 1.7 ** -6) < 1e-9
    assert R.OMEGA_MAX < 15.707963267948966          # DigitalAnalogDevice max_amp
    assert s.delta_final < 125.66370614359172        # max_abs_detuning
    assert s.T_ns % 4 == 0 and s.ramp_ns % 4 == 0
    om, de = s.samples()
    assert len(om) == len(de) == s.T_ns and om.max() <= R.OMEGA_MAX + 1e-12


def test_subspace_counts_and_normalization():
    inst = make_instance(0, "pilot")
    pos, w = inst.positions[:6], {i: inst.w[i] for i in range(6)}
    S0, S1 = R.Subspace(pos, w, 0), R.Subspace(pos, w, 1)
    assert len(S0.states) < len(S1.states)
    p = S1.evolve(R.Schedule(5, 2))
    assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


def test_single_atom_rabi_limit():
    # resonant constant drive would give Rabi oscillation; here just check probability bounds
    S = R.Subspace([(0, 0)], {0: 1.0})
    p = S.evolve(R.Schedule(10, 4))
    assert set(S.states.tolist()) == {0, 1} and 0 < p[S.states.tolist().index(1)] < 1
