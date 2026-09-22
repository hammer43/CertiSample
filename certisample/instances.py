"""A-G0 / C-G0 instance generator, exactly as specified in prereg/g0_spec.yaml."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class Instance:
    seed: int
    split: str
    positions: list          # lattice coordinates of occupied sites, in node order
    G: nx.Graph
    w: dict                  # normalized weights, max = 1


def make_instance(seed: int, split: str, side: int = 5, occupied: int = 20,
                  low: float = 0.5, high: float = 1.0) -> Instance:
    rng = np.random.default_rng(seed)
    sites = [(x, y) for x in range(side) for y in range(side)]
    idx = sorted(rng.choice(len(sites), size=occupied, replace=False).tolist())
    pos = [sites[i] for i in idx]
    G = nx.Graph()
    G.add_nodes_from(range(occupied))
    for i in range(occupied):
        for j in range(i + 1, occupied):
            if max(abs(pos[i][0] - pos[j][0]), abs(pos[i][1] - pos[j][1])) <= 1:
                G.add_edge(i, j)
    raw = rng.uniform(low, high, occupied)
    raw = raw / raw.max()
    return Instance(seed, split, pos, G, {i: float(raw[i]) for i in range(occupied)})


def seed_range(spec: dict, split: str) -> list[int]:
    lo, hi = spec["instances"]["seeds"][split]
    return list(range(lo, hi + 1))
