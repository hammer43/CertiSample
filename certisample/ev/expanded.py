"""Expanded network (model spec section 4).

IMPLEMENTATION NOTE — needs confirmation against the thesis equations before production.
The spec names four arc families; this module implements them as:
  (a) physical arcs i -> j for candidate-eligible physical nodes with sp(i,j) <= R_EV;
  (b) artificial-origin access arcs Oa_k -> j with sp(O_k, j) <= R_EV / 2   (half-range departure);
  (c) artificial-destination access arcs i -> Da_k with sp(i, D_k) <= R_EV / 2 (half-range arrival);
  (d) a direct arc Oa_k -> Da_k when sp(O_k, D_k) <= R_EV (never active for retained pairs,
      which satisfy D_k > R_EV by construction, but included for completeness).
Arc energy length is ell = sp / R_EV; artificial connectors carry ell = 0 only where the
underlying road distance is zero. Reaching a physical node j requires a station at j
(spec section 7 activation constraint); artificial nodes need no station.
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
        if net.sp.get(o, {}).get(d, float("inf")) <= R:
            A.append((("O", k), ("D", k), net.sp[o][d] / R))
        arcs[k] = A
    return Expanded(arcs, phys)
