"""Apple RoomPlan (CapturedRoom JSON) -> scan_map map dir, and RoomPlan <-> RGB-D TSDF fusion.

    python roomplan_map.py build captures/session_x_roomplan/roomplan.json maps/house_rp
    python roomplan_map.py align captures/session_x_roomplan/roomplan.json maps/house maps/house_fused

`build`: walls (and closed doors) -> obstacles, open doors and openings carved passable,
floor polygon -> free, objects -> obstacles + landmarks in places.json
({"xy", "kind": "landmark", "desc": category, "sim": {shape, size, pos, yaw_deg, rgba}}).
`align`: RoomPlan and the TSDF come from different AR sessions (different world frames,
both gravity-aligned), so a 2D rigid transform (yaw + xy) is found: brute-force yaw (1 deg)
x FFT cross-correlation for xy, then trimmed 2D ICP of RoomPlan wall points against the
TSDF map's obstacle cells. Output (RoomPlan frame): RoomPlan structure + TSDF obstacles /
free space + RoomPlan landmarks, alignment.json (yaw, t, RMSE, inlier fraction), alignment.png.

RoomPlan JSON: every `transform` is 16 floats, simd column-major (columns flattened, last 4 =
translation, 1); world is the ARKit session frame (Y-up). Surfaces (walls/doors/openings/
windows) have dimensions [width, height, 0] along their local x/y, centred on the transform.
Objects have dimensions [x, y, z] of a box centred on the transform. --closed-doors-passable
treats every door as an opening.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_map  # noqa: E402

RES = scan_map.RES
W_YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
WALL_THICK = 0.12
RGBA = {"refrigerator": [0.85, 0.85, 0.88, 1], "table": [0.55, 0.38, 0.22, 1], "chair": [0.3, 0.3, 0.35, 1],
        "storage": [0.6, 0.5, 0.4, 1], "sofa": [0.25, 0.4, 0.3, 1], "bed": [0.8, 0.8, 0.9, 1],
        "television": [0.1, 0.1, 0.1, 1], "sink": [0.8, 0.85, 0.9, 1], "stove": [0.3, 0.3, 0.3, 1],
        "oven": [0.3, 0.3, 0.3, 1], "dishwasher": [0.75, 0.75, 0.78, 1], "washerDryer": [0.9, 0.9, 0.9, 1],
        "toilet": [0.95, 0.95, 0.95, 1], "bathtub": [0.95, 0.95, 0.95, 1], "fireplace": [0.4, 0.2, 0.15, 1],
        "stairs": [0.5, 0.5, 0.5, 1]}


def _M(t):
    return np.array(t, float).reshape(4, 4).T     # column-major -> 4x4


def _cat(e):
    c = e.get("category", {})
    return next(iter(c)) if isinstance(c, dict) and c else str(c)


def load_roomplan(path: str) -> dict:
    """Everything in the Z-up map frame (floor at z=0) as 2D segments / footprints."""
    rp = json.load(open(path))
    # floor height (RoomPlan world is Y-up): floor surface, else lowest wall bottom
    if rp.get("floors"):
        floor_y = float(np.median([_M(f["transform"])[1, 3] for f in rp["floors"]]))
    else:
        floor_y = min(_M(w["transform"])[1, 3] - w["dimensions"][1] / 2 for w in rp["walls"])
    T = W_YUP_TO_ZUP.copy()
    T[2, 3] = -floor_y                                   # map frame: z = y_rp - floor_y

    def seg(e):
        M = T @ _M(e["transform"])
        c, ax = M[:3, 3], M[:3, 0]
        w, h = e["dimensions"][0], e["dimensions"][1]
        return {"a": (c - ax * w / 2)[:2], "b": (c + ax * w / 2)[:2], "z0": float(c[2] - h / 2),
                "z1": float(c[2] + h / 2), "cat": _cat(e), "id": e.get("identifier"),
                "parent": e.get("parentIdentifier"), "attrs": e.get("category", {})}

    out = {"T_rp_to_map": T, "floor_y": floor_y,
           "walls": [seg(w) for w in rp.get("walls", [])],
           "doors": [seg(d) for d in rp.get("doors", [])],
           "openings": [seg(o) for o in rp.get("openings", [])],
           "windows": [seg(o) for o in rp.get("windows", [])],
           "floors": [], "objects": []}
    for f in rp.get("floors", []):
        M = T @ _M(f["transform"])
        if f.get("polygonCorners"):
            P = np.c_[np.array(f["polygonCorners"], float), np.ones(len(f["polygonCorners"]))]
            out["floors"].append((M @ P.T)[:2].T)
        else:
            sx, sy = f["dimensions"][0] / 2, f["dimensions"][1] / 2
            P = np.array([[-sx, -sy, 0, 1], [sx, -sy, 0, 1], [sx, sy, 0, 1], [-sx, sy, 0, 1]])
            out["floors"].append((M @ P.T)[:2].T)
    for k, o in enumerate(rp.get("objects", [])):
        M = T @ _M(o["transform"])
        dx, dy, dz = o["dimensions"]
        # local box axes: x -> width, y -> height (up), z -> depth
        corners = np.array([[sx * dx / 2, 0, sz * dz / 2, 1] for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        foot = (M @ corners.T)[:2].T
        yaw = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        out["objects"].append({"cat": _cat(o), "center": M[:3, 3].tolist(), "dims": [dx, dz, dy],
                               "footprint": foot, "yaw_deg": yaw, "z0": float(M[2, 3] - dy / 2),
                               "z1": float(M[2, 3] + dy / 2),
                               "confidence": next(iter(o.get("confidence", {"?": 0})))})
    return out


def _bounds(rp, extra=None):
    pts = [s["a"] for s in rp["walls"]] + [s["b"] for s in rp["walls"]]
    pts += [p for f in rp["floors"] for p in f] + [p for o in rp["objects"] for p in o["footprint"]]
    P = np.array(pts)
    if extra is not None and len(extra):
        P = np.vstack([P, extra])
    lo, hi = P.min(0) - 0.5, P.max(0) + 0.5
    return lo, (int(math.ceil((hi[1] - lo[1]) / RES)), int(math.ceil((hi[0] - lo[0]) / RES)))


def rasterize(rp, origin, shape, closed_doors_passable=False):
    """-> floor, obstacle (bool HxW) for the RoomPlan model."""
    import cv2

    H, W = shape
    px = lambda p: np.round((np.asarray(p) - origin) / RES - 0.5).astype(np.int32)  # noqa: E731
    floor = np.zeros((H, W), np.uint8)
    for f in rp["floors"]:
        cv2.fillPoly(floor, [px(f).reshape(-1, 1, 2)], 1)
    obst = np.zeros((H, W), np.uint8)
    thick = max(1, int(round(WALL_THICK / RES)))
    band = scan_map.BAND
    for s in rp["walls"] + rp["windows"]:
        if s["z1"] > band[0] and s["z0"] < band[1]:      # windows above the robot don't block
            cv2.line(obst, tuple(px(s["a"])), tuple(px(s["b"])), 1, thick + 1)
    carve = rp["openings"] + [d for d in rp["doors"]
                              if closed_doors_passable or d["attrs"].get("door", {}).get("isOpen", False)]
    for s in carve:
        if s["z0"] < 0.3:                                   # reaches the floor -> walkable gap
            a, b = np.asarray(s["a"]), np.asarray(s["b"])
            u = (b - a) / max(1e-6, np.linalg.norm(b - a))
            a, b = a + u * 0.05, b - u * 0.05
            cv2.line(obst, tuple(px(a)), tuple(px(b)), 0, thick + 3)
            cv2.line(floor, tuple(px(a)), tuple(px(b)), 1, thick + 3)
    for o in rp["objects"]:
        if o["z1"] > band[0] and o["z0"] < band[1]:
            cv2.fillPoly(obst, [px(o["footprint"]).reshape(-1, 1, 2)], 1)
    return floor.astype(bool), obst.astype(bool)


def landmarks(rp) -> dict:
    places, count = {}, {}
    for o in rp["objects"]:
        c = o["cat"]
        count[c] = count.get(c, 0) + 1
        name = f"{c}_{count[c]}"
        places[name] = {"xy": [round(o["center"][0], 3), round(o["center"][1], 3)], "kind": "landmark",
                        "desc": c, "confidence": o["confidence"],
                        "sim": {"shape": "box", "size": [round(v, 3) for v in o["dims"]],  # full extents x,y,z
                                "pos": [round(v, 3) for v in o["center"]], "yaw_deg": round(o["yaw_deg"], 1),
                                "rgba": RGBA.get(c, [0.6, 0.6, 0.6, 1])}}
    return places


def write_map(out, free, obstacle, origin, meta_extra, places):
    os.makedirs(out, exist_ok=True)
    trav = ~scan_map._dilate(~free, int(round(scan_map.ROBOT_RADIUS / RES)))
    np.savez_compressed(os.path.join(out, "grid.npz"), free=free, obstacle=obstacle, traversable=trav)
    json.dump({"origin": [float(origin[0]), float(origin[1])], "res": RES, "floor_z": 0.0,
               "shape": list(free.shape), "up": "z", **meta_extra}, open(os.path.join(out, "meta.json"), "w"), indent=2)
    pf = os.path.join(out, "places.json")
    old = json.load(open(pf)) if os.path.exists(pf) else {}
    keep = {k: v for k, v in old.items() if v.get("kind") != "landmark"}   # keep user places (start, ...)
    json.dump({**places, **keep}, open(pf, "w"), indent=2)
    scan_map.render_map(out)


def build(rp_path, out, closed_doors_passable=False):
    rp = load_roomplan(rp_path)
    origin, shape = _bounds(rp)
    floor, obst = rasterize(rp, origin, shape, closed_doors_passable)
    free = floor & ~obst
    write_map(out, free, obst, origin, {"scan": os.path.abspath(rp_path), "source": "roomplan",
                                         "T_roomplan_to_map": rp["T_rp_to_map"].tolist()}, landmarks(rp))
    n_open = sum(d["attrs"].get("door", {}).get("isOpen", False) for d in rp["doors"])
    print(f"RoomPlan: {len(rp['walls'])} walls, {len(rp['doors'])} doors ({n_open} open), "
          f"{len(rp['openings'])} openings, {len(rp['objects'])} objects -> {out}/map.png "
          f"({free.sum() * RES * RES:.1f} m² free)")
    return rp


# ---------------- RoomPlan <-> TSDF alignment ----------------

def wall_points(rp, step=0.05):
    P = []
    for s in rp["walls"]:
        if s["z0"] > 0.5:                     # wall pieces above doors etc.
            continue
        a, b = np.asarray(s["a"]), np.asarray(s["b"])
        n = max(2, int(np.linalg.norm(b - a) / step))
        P.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(P)


def _rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def icp2d(src, dst, R, t, iters=60, thr=(0.6, 0.15)):
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    for k in range(iters):
        th = thr[0] + (thr[1] - thr[0]) * min(1.0, k / (iters * 0.6))
        p = src @ R.T + t
        d, j = tree.query(p)
        m = d < th
        if m.sum() < 20:
            break
        a, b = src[m], dst[j[m]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        Rn = (U @ Vt).T
        if np.linalg.det(Rn) < 0:
            Vt[-1] *= -1
            Rn = (U @ Vt).T
        R, t = Rn, cb - Rn @ ca
    p = src @ R.T + t
    d, _ = tree.query(p)
    m = d < thr[1]
    return R, t, float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("inf"), float(m.mean())


def align(rp, tsdf_map: str):
    """2D rigid transform taking RoomPlan map coords -> TSDF map coords."""
    from scipy import ndimage
    from scipy.signal import fftconvolve

    m = scan_map.Map(tsdf_map)
    r, c = np.nonzero(m.obstacle)
    dst = np.c_[m.ox + (c + 0.5) * RES, m.oy + (r + 0.5) * RES]
    src = wall_points(rp)
    cs = 0.10                                            # coarse search resolution
    lo = dst.min(0) - 1
    shp = (np.ceil((dst.max(0) + 1 - lo) / cs)).astype(int)[::-1]
    tgt = np.zeros(shp)
    ij = ((dst - lo) / cs).astype(int)
    tgt[ij[:, 1], ij[:, 0]] = 1
    tgt = ndimage.gaussian_filter(tgt, 1.5)
    sc = src.mean(0)
    best = []
    for deg in range(0, 360, 1):
        R = _rot(math.radians(deg))
        p = (src - sc) @ R.T
        plo = p.min(0)
        sshp = (np.ceil((p.max(0) - plo) / cs) + 1).astype(int)[::-1]
        img = np.zeros(sshp)
        q = ((p - plo) / cs).astype(int)
        img[q[:, 1], q[:, 0]] = 1
        corr = fftconvolve(tgt, img[::-1, ::-1], mode="full")
        k = np.unravel_index(np.argmax(corr), corr.shape)
        # corr index k corresponds to placing img's origin at (k - img.shape + 1) in tgt
        oy, ox = k[0] - sshp[0] + 1, k[1] - sshp[1] + 1
        t = lo + np.array([ox, oy]) * cs - plo - R @ sc
        best.append((corr[k], deg, R, t))
    best.sort(key=lambda x: -x[0])
    cands = []
    for score, deg, R, t in best[:6]:
        R2, t2, rmse, inl = icp2d(src, dst, R, t)
        cands.append((inl, -rmse, deg, R2, t2, rmse, score))
    cands.sort(key=lambda x: (round(x[0], 2), x[1]), reverse=True)
    inl, _, deg0, R, t, rmse, score = cands[0]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return {"R": R, "t": t, "yaw_deg": yaw, "rmse_m": rmse, "inlier_frac": inl, "coarse_yaw_deg": deg0,
            "n_wall_points": len(src), "n_tsdf_points": len(dst),
            "candidates": [{"coarse_yaw": c[2], "inlier_frac": round(c[0], 3), "rmse": round(c[5], 4)} for c in cands]}


def fuse(rp_path, tsdf_map, out, closed_doors_passable=False):
    """RoomPlan structure + TSDF detail in the RoomPlan map frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rp = load_roomplan(rp_path)
    al = align(rp, tsdf_map)
    R, t = al["R"], al["t"]                              # rp -> tsdf
    Ri, ti = R.T, -R.T @ t                               # tsdf -> rp
    m = scan_map.Map(tsdf_map)

    def cells_to_rp(mask):
        r, c = np.nonzero(mask)
        sub = np.array([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]]) * RES
        P = np.c_[m.ox + (c + 0.5) * RES, m.oy + (r + 0.5) * RES]
        P = (P[:, None, :] + sub[None]).reshape(-1, 2)
        return P @ Ri.T + ti

    t_obs, t_free = cells_to_rp(m.obstacle), cells_to_rp(m.free)
    origin, shape = _bounds(rp, np.vstack([t_obs, t_free]))
    floor, obst = rasterize(rp, origin, shape, closed_doors_passable)

    def splat(P):
        g = np.zeros(shape, bool)
        ij = np.floor((P - origin) / RES).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
        g[ij[ok, 1], ij[ok, 0]] = True
        return g

    tobs, tfree = splat(t_obs), splat(t_free)
    obstacle = obst | tobs
    free = (floor | tfree) & ~obstacle
    al_out = {"yaw_deg": round(al["yaw_deg"], 3), "t": np.round(al["t"], 4).tolist(),
              "rmse_m": round(al["rmse_m"], 4), "inlier_frac": round(al["inlier_frac"], 3),
              "inlier_threshold_m": 0.15, "direction": "roomplan map -> TSDF map: p_tsdf = R p_rp + t",
              "R": np.round(R, 6).tolist(), "n_wall_points": al["n_wall_points"],
              "n_tsdf_obstacle_cells": al["n_tsdf_points"], "candidates": al["candidates"]}
    write_map(out, free, obstacle, origin,
              {"scan": os.path.abspath(rp_path), "source": "roomplan+tsdf", "tsdf_map": os.path.abspath(tsdf_map),
               "T_roomplan_to_map": rp["T_rp_to_map"].tolist(), "alignment": al_out}, landmarks(rp))
    json.dump(al_out, open(os.path.join(out, "alignment.json"), "w"), indent=2)
    # overlay figure
    wp = wall_points(rp)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(*t_obs[::4].T, ",", color="#555", alpha=0.5, label="TSDF obstacles (cleaned map)")
    for s in rp["walls"]:
        ax.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]], "-", color="#d0453a", lw=1.5)
    for s in rp["doors"] + rp["openings"]:
        ax.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]], "-", color="#2a7d4f", lw=3)
    for o in rp["objects"]:
        f = np.vstack([o["footprint"], o["footprint"][:1]])
        ax.plot(*f.T, "-", color="#4a90d9", lw=1)
    ax.plot([], [], "-", color="#d0453a", label="RoomPlan walls")
    ax.plot([], [], "-", color="#2a7d4f", lw=3, label="doors / openings")
    ax.plot([], [], "-", color="#4a90d9", label="RoomPlan objects")
    ax.set_title(f"RoomPlan <-> TSDF: yaw {al['yaw_deg']:.1f} deg, RMSE {al['rmse_m'] * 100:.1f} cm, "
                 f"{al['inlier_frac'] * 100:.0f}% of wall points within 15 cm")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "alignment.png"), dpi=110)
    plt.close(fig)
    print(f"aligned: yaw {al['yaw_deg']:.2f} deg, t {np.round(al['t'], 2)}, RMSE {al['rmse_m'] * 100:.1f} cm, "
          f"inliers {al['inlier_frac'] * 100:.0f}% -> {out}/map.png, alignment.png")
    return al_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("roomplan"); p.add_argument("map_dir")
    p.add_argument("--closed-doors-passable", action="store_true")
    p = sub.add_parser("align"); p.add_argument("roomplan"); p.add_argument("tsdf_map_dir"); p.add_argument("out")
    p.add_argument("--closed-doors-passable", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.roomplan, a.map_dir, a.closed_doors_passable)
    else:
        fuse(a.roomplan, a.tsdf_map_dir, a.out, a.closed_doors_passable)


if __name__ == "__main__":
    main()
