"""Instance generator (model spec v1.4.1, sections 2-4, 6, 8)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from . import tntp

SITE_SEED, OD_SEED, DEMAND_SEED = 310000, 320000, 330000
CANDIDATE_FRACTION = 0.80
DEMAND_SIGMA = 0.25
RHO_GRID = (0.75, 1.00, 1.25, 1.50, 2.00, 2.50)
PHI_GRID = (1.0, 10.0, 100.0)
PHI_SOURCE = 1e6


@dataclass
class Network:
    name: str
    G: nx.DiGraph
    trips: dict
    coords_norm: dict
    s_nn: float
    sp: dict                    # shortest-path distances on the road network
    R_EV: float
    K_plus: list                # OD pairs with demand > 0 and distance > R_EV

    @property
    def nodes(self):
        return sorted(self.G.nodes())


def load_network(name, net_path, trips_path, node_path, geographic=True) -> Network:
    G = tntp.read_net(net_path)
    trips = tntp.read_trips(trips_path)
    coords, s_nn = tntp.normalize(tntp.project(tntp.read_nodes(node_path), geographic))
    sp = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    dist = [sp[o][d] for (o, d), f in trips.items()
            if o != d and f > 0 and d in sp.get(o, {})]
    R_EV = float(np.median(dist))                                   # spec 2.3
    K_plus = sorted((o, d) for (o, d), f in trips.items()
                    if o != d and f > 0 and sp.get(o, {}).get(d, math.inf) > R_EV)
    return Network(name, G, trips, coords, s_nn, sp, R_EV, K_plus)


@dataclass
class Instance:
    j: int
    net: Network
    sites: list                 # candidate sites C_j
    od: list                    # selected OD pairs
    demand: dict                # d_k, mean 1
    fbar: dict = field(default_factory=dict)

    def conflict_graph(self, rho: float) -> nx.Graph:
        """Spec 2.4: strict inequality, normalized coordinates."""
        H = nx.Graph()
        H.add_nodes_from(self.sites)
        c = self.net.coords_norm
        for a, i in enumerate(self.sites):
            for jj in self.sites[a + 1:]:
                if math.dist(c[i], c[jj]) < rho:
                    H.add_edge(i, jj)
        return H

    def opening_costs(self, phi: float) -> dict:
        """Spec 6 and 8: f_i = phi * dbar (dbar = 1), fbar_i = f_i - half OD demand."""
        f = {i: phi for i in self.sites}
        for (o, d), dk in self.demand.items():
            if o in f:
                f[o] -= 0.5 * dk
            if d in f:
                f[d] -= 0.5 * dk
        return f


def make_instance(net: Network, j: int, k_target: int) -> Instance:
    nodes = net.nodes
    m = math.floor(CANDIDATE_FRACTION * len(nodes))
    rng_site = np.random.default_rng(SITE_SEED + j)
    sites = sorted(int(x) for x in rng_site.choice(nodes, size=m, replace=False))

    rng_od = np.random.default_rng(OD_SEED + j)
    kp = net.K_plus
    w = np.array([net.trips[k] for k in kp], float)
    n_take = min(k_target, len(kp))
    idx = rng_od.choice(len(kp), size=n_take, replace=False, p=w / w.sum())
    od = sorted(kp[int(i)] for i in idx)

    rng_d = np.random.default_rng(DEMAND_SEED + j)
    base = np.array([net.trips[k] for k in od], float)
    wk = base / base.mean()
    z = rng_d.standard_normal(len(od))
    dt = wk * np.exp(DEMAND_SIGMA * z - 0.5 * DEMAND_SIGMA ** 2)
    dk = dt / dt.mean()
    return Instance(j, net, sites, od, {k: float(v) for k, v in zip(od, dk)})
