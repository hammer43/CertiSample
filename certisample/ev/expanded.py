"""Expanded network (model spec section 4; Kınay thesis §2.2.2).

The thesis construction has A1: Oa_k -> O_k (zero length), A2: D_k -> Da_k
(zero length), A3 physical i -> j when shortest-path distance <= R_EV, and
half-range A4/A5 access/egress arcs at R_EV/2.

CertiSample fixes y_j = 0 outside the sampled candidate-site set, so such physical
nodes cannot be charging stops. This module therefore uses the exact projected
network on candidate-eligible physical nodes. For candidate origins/destinations,
the zero-length A1/A2 connectors are already the zero-distance cases of A4/A5.
No direct Oa_k -> Da_k arc is part of the source construction.

Arc energy length is ell = shortest-path distance / R_EV. Reaching a physical
head node requires an open station there; artificial destination nodes do not.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Expanded:
    """Per-OD-pair expanded arcs: (tail, head, ell). Tails/heads are ints (physical)
    or ('O', k) / ('D', k) for the artificial nodes of pair k."""
    arcs: dict          # k -> list of (tail, head, ell)
    physical: list      # physical nodes usable as charging stops


def build(inst, half_range: float = 0.5) -> Expanded:
    net = inst.net
    R = net.R_EV
    phys = sorted(set(inst.sites))
    arcs = {}
    for k in inst.od:
        o, d = k
        A = []
        for i in phys:
            if net.sp.get(o, {}).get(i, float("inf")) <= half_range * R:
                A.append((("O", k), i, net.sp[o][i] / R))
            if net.sp.get(i, {}).get(d, float("inf")) <= half_range * R:
                A.append((i, ("D", k), net.sp[i][d] / R))
            for j in phys:
                if i != j and net.sp.get(i, {}).get(j, float("inf")) <= R:
                    A.append((i, j, net.sp[i][j] / R))
        arcs[k] = A
    return Expanded(arcs, phys)
