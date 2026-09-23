"""TNTP readers (TransportationNetworks repository format)."""
from __future__ import annotations

import math
import re
from pathlib import Path

import networkx as nx
import numpy as np


def read_net(path) -> nx.DiGraph:
    """Directed road network with the TNTP `length` field as arc length."""
    text = Path(path).read_text(encoding="utf-8")
    body = text.split("<END OF METADATA>", 1)[1]
    G = nx.DiGraph()
    for line in body.splitlines():
        line = line.strip().rstrip(";").strip()
        if not line or line.startswith("~"):
            continue
        f = line.split()
        u, v, length = int(f[0]), int(f[1]), float(f[4])
        G.add_edge(u, v, length=length)
    return G


def read_trips(path) -> dict:
    """{(origin, destination): flow} for strictly positive flows."""
    text = Path(path).read_text(encoding="utf-8")
    body = text.split("<END OF METADATA>", 1)[1]
    trips, origin = {}, None
    for chunk in re.split(r"(Origin\s+\d+)", body):
        m = re.match(r"Origin\s+(\d+)", chunk.strip())
        if m:
            origin = int(m.group(1))
            continue
        if origin is None:
            continue
        for d, f in re.findall(r"(\d+)\s*:\s*([0-9.eE+-]+)", chunk):
            flow = float(f)
            if flow > 0:
                trips[(origin, int(d))] = flow
    return trips


def read_nodes(path) -> dict:
    """{node: (x, y)} as supplied (geographic or planar)."""
    text = Path(path).read_text(encoding="utf-8")
    out = {}
    for line in text.splitlines()[1:]:
        f = line.strip().rstrip(";").split()
        if len(f) >= 3:
            try:
                out[int(f[0])] = (float(f[1]), float(f[2]))
            except ValueError:
                continue
    return out


def project(coords: dict, geographic: bool = True) -> dict:
    """Spec 2.4: local equirectangular projection about the centroid for lon/lat input."""
    if not geographic:
        return dict(coords)
    lon = np.array([c[0] for c in coords.values()])
    lat = np.array([c[1] for c in coords.values()])
    lat0 = math.radians(float(lat.mean()))
    R = 6371.0088  # km
    return {n: (math.radians(c[0]) * R * math.cos(lat0), math.radians(c[1]) * R)
            for n, c in coords.items()}


def normalize(coords: dict) -> tuple[dict, float]:
    """Spec 2.4: divide by the median nearest-neighbour distance s_NN."""
    ids = sorted(coords)
    P = np.array([coords[i] for i in ids], float)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    s_nn = float(np.median(D.min(1)))
    return {i: tuple(P[k] / s_nn) for k, i in enumerate(ids)}, s_nn
