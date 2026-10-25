"""Fused RGB-D scan -> cleaned static map for the K1 (people and clutter removed).

    python rgbd_ingest.py captures/session_x scans/house --label
    python map_clean.py scans/house maps/house          # -> maps/house/map.png, grid.npz, ...
    python scan_map.py place maps/house start 0.4 0.3 --yaw 90   # then plan / rehearse as usual

Steps: RANSAC floor plane -> level so the floor is z=0 -> crop z to [-0.05, 2.2] ->
statistical outlier removal -> drop small isolated connected pieces -> delete everything inside
clutter/person boxes from objects.json (expanded 10 cm, walls protected) and patch the
floor under them -> clean.ply (a triangle mesh, so scan_map.build can sample it) ->
scan_map.build(clean.ply, map_dir, up="z").

Outputs in <map_dir>: clean.ply, clean_points.ply, leveled_raw.ply, semistatic.json
(movable furniture boxes, map frame), clean_report.json, before_after.png, plus
scan_map's grid.npz / meta.json / places.json / map.png. "before" grid -> <map_dir>/_before/.
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

Z_MIN, Z_MAX = -0.05, 2.2
BOX_PAD = 0.10
FLOOR_KEEP = 0.04            # never delete points this close to the floor inside object boxes
SMALL_VOL = 0.02             # m^3: connected pieces smaller than this are noise
SMALL_TRIS = 30
REMOVE_CATS = ("clutter", "person")


# ---------------- occupancy grid straight from points (fallback + live preview) ----------------

def grid_from_points(pts: np.ndarray, res: float = scan_map.RES, band=scan_map.BAND, floor_z=None):
    """Same logic as scan_map.build, but on a point cloud (no mesh sampling)."""
    from scipy import ndimage

    z = pts[:, 2]
    if floor_z is None:
        low = z[z < np.percentile(z, 30)]
        hist, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.04, 0.02))
        floor_z = float(edges[np.argmax(hist)] + 0.01)
    xmin, ymin = pts[:, 0].min() - 0.5, pts[:, 1].min() - 0.5
    W = int(math.ceil((pts[:, 0].max() + 0.5 - xmin) / res))
    H = int(math.ceil((pts[:, 1].max() + 0.5 - ymin) / res))
    ix = np.clip(((pts[:, 0] - xmin) / res).astype(int), 0, W - 1)
    iy = np.clip(((pts[:, 1] - ymin) / res).astype(int), 0, H - 1)
    h = z - floor_z
    floor = np.zeros((H, W), bool)
    sel = np.abs(h) < 0.06
    floor[iy[sel], ix[sel]] = True
    floor = ndimage.binary_closing(floor, iterations=3)
    counts = np.zeros((H, W), int)
    sel = (h > band[0]) & (h < band[1])
    np.add.at(counts, (iy[sel], ix[sel]), 1)
    obstacle = counts >= 2
    free = floor & ~obstacle
    return {"free": free, "obstacle": obstacle, "origin": [float(xmin), float(ymin)], "res": res,
            "floor_z": floor_z}


def build_from_points(pts: np.ndarray, out: str, scan_path: str):
    """Fallback writing the same grid.npz / meta.json / places.json as scan_map.build."""
    g = grid_from_points(pts)
    trav = ~scan_map._dilate(~g["free"], int(round(scan_map.ROBOT_RADIUS / scan_map.RES)))
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "grid.npz"), free=g["free"], obstacle=g["obstacle"], traversable=trav)
    json.dump({"origin": g["origin"], "res": scan_map.RES, "floor_z": g["floor_z"], "shape": list(g["free"].shape),
               "scan": os.path.abspath(scan_path), "up": "z"}, open(os.path.join(out, "meta.json"), "w"), indent=2)
    if not os.path.exists(os.path.join(out, "places.json")):
        json.dump({}, open(os.path.join(out, "places.json"), "w"))
    scan_map.render_map(out)


def build_map(ply: str, out: str, pts_fallback: np.ndarray):
    try:
        scan_map.build(ply, out, up="z")
        return "scan_map.build"
    except Exception as e:  # noqa: BLE001
        print(f"warning: scan_map.build failed on {ply} ({e}); building the grid from points", file=sys.stderr)
        build_from_points(pts_fallback, out, ply)
        return "map_clean.build_from_points"


# ---------------- geometry helpers ----------------

def _rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def find_floor(pts: np.ndarray, max_planes=8, dist=0.03):
    """Lowest large near-horizontal RANSAC plane. Returns (normal pointing up, d, n_inliers)."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc = pc.voxel_down_sample(0.04)
    best = None
    rest = pc
    total = len(pc.points)
    for _ in range(max_planes):
        if len(rest.points) < 200:
            break
        (a, b, c, d), inl = rest.segment_plane(dist, 3, 1000)
        n = np.array([a, b, c])
        if n[2] < 0:
            n, d = -n, -d
        P = np.asarray(rest.points)[inl]
        if n[2] > math.cos(math.radians(15)) and len(inl) > 0.03 * total:
            z = float(np.median(P[:, 2]))
            if best is None or z < best[3] - 0.3:      # a clearly lower big plane wins
                best = (n, d, len(inl), z)
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        sys.exit("no floor plane found (no large plane with normal within 15 deg of +Z)")
    return best[0], best[1], best[2]


def wall_planes(pts: np.ndarray, max_planes=14, dist=0.04):
    """Big vertical planes (>= 1 m tall and >= 1 m long): never deleted by object boxes."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.05)
    rest, out = pc, []
    for _ in range(max_planes):
        if len(rest.points) < 200:
            break
        (a, b, c, d), inl = rest.segment_plane(dist, 3, 1000)
        P = np.asarray(rest.points)[inl]
        n = np.array([a, b, c])
        if abs(n[2]) < 0.15 and np.ptp(P[:, 2]) > 1.0 and np.hypot(*np.ptp(P[:, :2], axis=0)) > 1.0:
            out.append((n, d, P[:, :2].min(0) - 0.1, P[:, :2].max(0) + 0.1))
        rest = rest.select_by_index(inl, invert=True)
    return out


def transform_box(center, extent, T):
    c, e = np.asarray(center), np.asarray(extent) / 2
    corners = np.array([[c[0] + sx * e[0], c[1] + sy * e[1], c[2] + sz * e[2], 1]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    w = (T @ corners.T)[:3].T
    return w.min(0), w.max(0)


def floor_patch(mn, mx, z=0.0):
    import open3d as o3d

    v = np.array([[mn[0], mn[1], z], [mx[0], mn[1], z], [mx[0], mx[1], z], [mn[0], mx[1], z]])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
    m.vertex_colors = o3d.utility.Vector3dVector(np.full((4, 3), 0.5))
    return m


# ---------------- cleanup ----------------

def clean(ingest_dir: str, map_dir: str, objects_path: str | None = None):
    import open3d as o3d

    os.makedirs(map_dir, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(os.path.join(ingest_dir, "fused.ply"))
    if len(mesh.triangles) == 0:
        sys.exit(f"{ingest_dir}/fused.ply has no triangles")
    objects_path = objects_path or os.path.join(ingest_dir, "objects.json")
    objects = json.load(open(objects_path)).get("objects", []) if os.path.exists(objects_path) else []
    if not os.path.exists(objects_path):
        print(f"note: no {objects_path}: no semantic removal (run rgbd_ingest.py --label)")

    # 1. floor -> z=0
    V = np.asarray(mesh.vertices)
    n, d, n_in = find_floor(V)
    R = _rot_a_to_b(n, np.array([0, 0, 1.0]))
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = d  # plane n.x + d = 0 -> after rotation z = -d
    mesh.transform(T)
    tilt = math.degrees(math.acos(min(1.0, n[2])))
    print(f"floor plane: normal tilt {tilt:.2f} deg, {n_in} inliers; levelled to z=0")
    o3d.io.write_triangle_mesh(os.path.join(map_dir, "leveled_raw.ply"), mesh)
    stats = {"floor_tilt_deg": round(tilt, 3), "floor_offset_m": round(float(d), 3),
             "vertices_in": len(mesh.vertices)}

    # 2. height crop
    V = np.asarray(mesh.vertices)
    drop = (V[:, 2] < Z_MIN) | (V[:, 2] > Z_MAX)
    stats["dropped_height"] = int(drop.sum())
    # 3. statistical outliers
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    _, keep = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
    out_mask = np.ones(len(V), bool)
    out_mask[keep] = False
    stats["dropped_outliers"] = int((out_mask & ~drop).sum())
    drop |= out_mask
    mesh.remove_vertices_by_mask(drop)

    # 4. small connected pieces (that aren't floor patches)
    tc, ntri, _ = mesh.cluster_connected_triangles()
    tc, ntri = np.asarray(tc), np.asarray(ntri)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    small = []
    for k in range(len(ntri)):
        sel = tc == k
        P = V[np.unique(F[sel])]
        vol = float(np.prod(np.maximum(np.ptp(P, axis=0), 0.01)))
        on_floor = P[:, 2].max() < 0.08
        if (vol < SMALL_VOL or ntri[k] < SMALL_TRIS) and not on_floor:
            small.append(k)
    # a small piece touching (< 10 cm) big geometry is a fragment of it (e.g. a counter top
    # seen at a grazing angle), not floating noise: keep it
    from scipy.spatial import cKDTree

    is_small = np.isin(tc, small)
    big_v = np.unique(F[~is_small])
    big_v = big_v[V[big_v, 2] > 0.08]        # touching only the floor doesn't count (ghosts stand on it)
    tree = cKDTree(V[big_v]) if len(big_v) else None
    rm_tri = np.zeros(len(F), bool)
    n_small = 0
    for k in small:
        sel = tc == k
        if tree is not None and tree.query(V[np.unique(F[sel])], distance_upper_bound=0.10)[0].min() < 0.10:
            continue
        rm_tri |= sel
        n_small += 1
    mesh.remove_triangles_by_mask(rm_tri)
    mesh.remove_unreferenced_vertices()
    stats["small_components_removed"] = n_small

    # 5. objects: delete clutter / person boxes, keep movable furniture as semi-static
    walls = wall_planes(np.asarray(mesh.vertices))
    V = np.asarray(mesh.vertices)
    protect = V[:, 2] < FLOOR_KEEP
    for wn, wd, lo, hi in walls:
        near = np.abs(V @ wn + wd) < 0.05
        inside = np.all((V[:, :2] >= lo) & (V[:, :2] <= hi), axis=1)
        protect |= near & inside
    obj_rm = np.zeros(len(V), bool)
    removed, semistatic, patches = [], [], []
    for o in objects:
        mn, mx = transform_box(o["center"], o["extent"], T)
        if o["category"] in REMOVE_CATS:
            mn_p, mx_p = mn - BOX_PAD, mx + BOX_PAD
            inside = np.all((V >= mn_p) & (V <= mx_p), axis=1)
            hit = inside & ~protect
            obj_rm |= hit
            removed.append({"label": o["label"], "category": o["category"], "min": mn_p.round(3).tolist(),
                            "max": mx_p.round(3).tolist(), "points_removed": int(hit.sum()),
                            "frames": o.get("frames", [])})
            if mn[2] < 0.3:                 # it stood on the floor: the floor continues under it
                patches.append(floor_patch(mn_p, mx_p))
        elif o["category"] == "movable_furniture":
            semistatic.append({"label": o["label"], "center": ((mn + mx) / 2).round(3).tolist(),
                               "extent": (mx - mn).round(3).tolist(), "min": mn.round(3).tolist(),
                               "max": mx.round(3).tolist(), "frames": o.get("frames", []),
                               "n_obs": o.get("n_obs", 1)})
    mesh.remove_vertices_by_mask(obj_rm)
    for p in patches:
        mesh += p
    mesh.remove_unreferenced_vertices()
    stats.update({"walls_protected": len(walls), "object_points_removed": int(obj_rm.sum()),
                  "vertices_out": len(mesh.vertices), "triangles_out": len(mesh.triangles)})

    clean_ply = os.path.join(map_dir, "clean.ply")
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(clean_ply, mesh)
    pts = np.asarray(mesh.sample_points_uniformly(max(50_000, int(mesh.get_surface_area() * 2500))).points)
    pts = np.vstack([pts, np.asarray(mesh.vertices)])
    o3d.io.write_point_cloud(os.path.join(map_dir, "clean_points.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    json.dump({"frame": "map (floor z=0, Z-up)", "objects": semistatic},
              open(os.path.join(map_dir, "semistatic.json"), "w"), indent=2)

    # 6. grids: after (the real map) and before (levelled, uncleaned) for the comparison
    stats["grid_builder"] = build_map(clean_ply, map_dir, pts)
    before_dir = os.path.join(map_dir, "_before")
    raw = o3d.io.read_triangle_mesh(os.path.join(map_dir, "leveled_raw.ply"))
    build_map(os.path.join(map_dir, "leveled_raw.ply"), before_dir, np.asarray(raw.vertices))
    stats.update(before_after(before_dir, map_dir, removed, semistatic))
    stats["removed_objects"] = removed
    stats["semistatic_objects"] = len(semistatic)
    stats["level_transform"] = T.round(6).tolist()
    json.dump(stats, open(os.path.join(map_dir, "clean_report.json"), "w"), indent=2)
    print(f"removed {len(removed)} clutter/person boxes ({stats['object_points_removed']} verts), "
          f"{n_small} small pieces, {stats['dropped_outliers']} outliers; "
          f"{stats['occupied_removed_pct']:.1f}% of occupied cells cleared -> {map_dir}/before_after.png")
    return stats


def before_after(before_dir: str, after_dir: str, removed: list, semistatic: list) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy import ndimage

    b, a = scan_map.Map(before_dir), scan_map.Map(after_dir)
    res = scan_map.RES
    # resample "after" obstacles into the "before" grid (origins differ by < a few cells)
    rb, cb = np.nonzero(b.obstacle)
    xs, ys = b.ox + (cb + 0.5) * res, b.oy + (rb + 0.5) * res
    ra, ca = ((ys - a.oy) / res).astype(int), ((xs - a.ox) / res).astype(int)
    ok = (ra >= 0) & (ra < a.obstacle.shape[0]) & (ca >= 0) & (ca < a.obstacle.shape[1])
    a_near = ndimage.binary_dilation(a.obstacle, iterations=1)   # 1-cell tolerance for resampling
    still = np.zeros(len(rb), bool)
    still[ok] = a_near[ra[ok], ca[ok]]
    n_before, n_after = int(b.obstacle.sum()), int(a.obstacle.sum())
    gone = int((~still).sum())
    pct = 100.0 * gone / max(1, n_before)
    cats = {c: sum(r["category"] == c for r in removed) for c in REMOVE_CATS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, m, title in ((axes[0], b, f"before: {n_before} occupied cells"),
                         (axes[1], a, f"after: {n_after} occupied cells ({pct:.1f}% of 'before' cleared)")):
        H, W = m.free.shape
        img = np.full((H, W, 3), 0.55)
        img[m.free] = (0.96, 0.96, 0.94)
        img[m.free & ~m.trav] = (0.85, 0.88, 0.95)
        img[m.obstacle] = (0.2, 0.2, 0.25)
        ax.imshow(img, origin="lower", extent=[m.ox, m.ox + W * res, m.oy, m.oy + H * res], interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, color="#4a90d9", alpha=0.25, lw=0.5)
    for r in removed:
        mn, mx = r["min"], r["max"]
        for ax in axes:
            ax.add_patch(Rectangle(mn[:2], mx[0] - mn[0], mx[1] - mn[1], fill=False, ec="#d0453a", lw=1.5,
                                   ls="-" if ax is axes[0] else "--"))
        axes[0].annotate(r["label"], mn[:2], fontsize=7, color="#d0453a")
    for s in semistatic:
        mn, mx = s["min"], s["max"]
        axes[1].add_patch(Rectangle(mn[:2], mx[0] - mn[0], mx[1] - mn[1], fill=False, ec="#2a7d4f", lw=1.2))
        axes[1].annotate(s["label"], mn[:2], fontsize=7, color="#2a7d4f")
    fig.suptitle(f"map cleanup: removed {cats['person']} person + {cats['clutter']} clutter objects "
                 f"(red), {len(semistatic)} semi-static furniture (green)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(after_dir, "before_after.png"), dpi=110)
    plt.close(fig)
    return {"occupied_cells_before": n_before, "occupied_cells_after": n_after,
            "occupied_cells_cleared": gone, "occupied_removed_pct": round(pct, 2),
            "removed_person": cats["person"], "removed_clutter": cats["clutter"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ingest_dir")
    ap.add_argument("map_dir")
    ap.add_argument("--objects", help="objects.json to use instead of <ingest_dir>/objects.json")
    a = ap.parse_args()
    clean(a.ingest_dir, a.map_dir, a.objects)


if __name__ == "__main__":
    main()
