import numpy as np

from certisample.a_g0_test import verdict


def test_verdict_pass_requires_both():
    assert verdict(np.full(30, 0.05), np.full(30, 0.05), 2000, 0, 0.95)["decision"] == "PASS"
    assert verdict(np.full(30, 0.05), np.full(30, -0.05), 2000, 0, 0.95)["decision"] == "FAIL"
    assert verdict(np.full(30, -0.05), np.full(30, 0.05), 2000, 0, 0.95)["decision"] == "FAIL"


def test_verdict_fails_when_ci_straddles_zero():
    d = np.r_[np.full(15, 0.1), np.full(15, -0.09)]
    assert verdict(d, np.full(30, 0.1), 5000, 0, 0.95)["decision"] == "FAIL"
