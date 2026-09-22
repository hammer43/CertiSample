import numpy as np

from certisample import c_g0
from certisample.rydberg import OMEGA_MAX, Subspace


def test_cold_schedule_shape():
    om, de = c_g0.samples(c_g0.COLD, 1000)
    assert om[0] == 0 and om[-1] == 0 and abs(om.max() - OMEGA_MAX) < 1e-9
    assert abs(de[0] + 4 * OMEGA_MAX) < 1e-9 and abs(de[-1] - 4 * OMEGA_MAX) < 1e-9
    assert np.all(np.diff(de) >= -1e-9)          # cold detuning is monotone


def test_parameters_are_clipped():
    om, de = c_g0.samples(np.array([5, 5, 5, 5, 50, 50, 50, 50.0]), 500)
    assert om.max() <= OMEGA_MAX + 1e-9 and de.max() <= 6 * OMEGA_MAX + 1e-9


def test_infeasible_samples_score_zero_and_target():
    inst = c_g0.instance(2000, "pilot")
    h = {i: inst.w[i] for i in range(len(inst.positions))}
    S = Subspace(inst.positions, h)
    obj = c_g0.Objective(S, inst.G, h, 400)
    assert obj.E.max() == 0.0 and obj.E.min() == obj.E_star < 0
    assert obj.E_star < obj.target < obj.E_cold        # strictly between optimum and cold


def test_lagrangian_deltas_positive():
    d = c_g0.lagrangian_deltas(2000, rounds=5)
    assert len(d) == 5 and all(x > 0 for x in d)          # budget constraints bind
