"""TNTP data fetch for the frozen networks (spec section 2.1).

Pinned repository: bstabler/TransportationNetworks
Commit used for the reference run: d1639b4ef218c17928ba573e806ddf8ba5e7ae6d
Record the commit actually used in the run manifest.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

COMMIT = "d1639b4ef218c17928ba573e806ddf8ba5e7ae6d"
BASE = "https://raw.githubusercontent.com/bstabler/TransportationNetworks"
FILES = {"SiouxFalls": ["SiouxFalls_net.tntp", "SiouxFalls_trips.tntp", "SiouxFalls_node.tntp"],
         "Anaheim": ["Anaheim_net.tntp", "Anaheim_trips.tntp", "Anaheim_node.tntp"],
         "Chicago-Sketch": ["ChicagoSketch_net.tntp", "ChicagoSketch_trips.tntp",
                            "ChicagoSketch_node.tntp"]}


def fetch(network: str, out_dir="data/tntp", commit: str = COMMIT) -> dict:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for f in FILES[network]:
        p = d / f
        if not p.exists():
            urllib.request.urlretrieve(f"{BASE}/{commit}/{network}/{f}", p)
        paths[f.split("_")[-1].replace(".tntp", "")] = p
    return paths
