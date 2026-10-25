"""iPhone LiDAR scan -> floor plan -> route -> rehearsal in the MuJoCo K1 twin.

Export the scan from the iPhone app (3D Scanner App / Polycam LiDAR mode) as
OBJ, PLY or GLB, then:

    python scan_map.py build scans/agihouse.obj maps/agihouse          # floor plan
    python scan_map.py place maps/agihouse start 0.4 0.3 --yaw 90      # where the K1 starts
    python scan_map.py place maps/agihouse kitchen 6.2 2.5             # named places
    python scan_map.py plan maps/agihouse kitchen                      # A* route -> waypoints
    python scan_map.py rehearse maps/agihouse kitchen --gif            # walk it in the twin

Read place coordinates off maps/<name>/map.png (1 m grid). `synth` writes a
fake Y-up room scan so the pipeline can be tested before the real scan exists.

Frames: the "map" frame is the scan's floor plane (x, y, z-up). The sim spawns
the K1 at its origin facing +x, so map -> sim uses the "start" place's pose.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = 0.05            # m per grid cell
ROBOT_RADIUS = 0.30   # inflation for planning (K1 shoulder half-width + margin)
BODY_RADIUS = 0.15    # tighter radius for the "did we hit something" check
BAND = (0.15, 1.10)   # obstacle heights above the floor that block a 0.95 m robot


# ---------------- scan -> occupancy grid ----------------

def load_points(path: str, up: str) -> np.ndarray:
    import trimesh

    if path.lower().endswith(".usdz"):
        sys.exit("USDZ isn't supported: export OBJ, PLY or GLB from the scanner app.")
    mesh = trimesh.load(path, force="mesh")
    n = int(min(3_000_000, max(200_000, mesh.area * 2500)))  # ~1 point per 20x20 mm
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = np.vstack([pts, mesh.vertices])
    if up == "y":  # ARKit / iPhone exports are Y-up: rotate to Z-up
        pts = np.column_stack([pts[:, 0], -pts[:, 2], pts[:, 1]])
    return pts


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    from scipy import ndimage

    r = radius_cells
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(mask, structure=(xx * xx + yy * yy) <= r * r)


def build(scan: str, out: str, up: str = "y"):
    from scipy import ndimage

    pts = load_points(scan, up)
    z = pts[:, 2]
    low = z[z < np.percentile(z, 30)]
    hist, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.02, 0.02))
    floor_z = float(edges[np.argmax(hist)] + 0.01)

    xmin, ymin = pts[:, 0].min() - 0.5, pts[:, 1].min() - 0.5
    W = int(math.ceil((pts[:, 0].max() + 0.5 - xmin) / RES))
    H = int(math.ceil((pts[:, 1].max() + 0.5 - ymin) / RES))
    ix = np.clip(((pts[:, 0] - xmin) / RES).astype(int), 0, W - 1)
    iy = np.clip(((pts[:, 1] - ymin) / RES).astype(int), 0, H - 1)
    h = z - floor_z

    floor = np.zeros((H, W), bool)
    sel = np.abs(h) < 0.06
    floor[iy[sel], ix[sel]] = True
    floor = ndimage.binary_closing(floor, iterations=3)  # fill LiDAR holes

    counts = np.zeros((H, W), int)
    sel = (h > BAND[0]) & (h < BAND[1])
    np.add.at(counts, (iy[sel], ix[sel]), 1)
    obstacle = counts >= 2  # ignore single stray points

    free = floor & ~obstacle
    traversable = ~_dilate(~free, int(round(ROBOT_RADIUS / RES)))

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "grid.npz"), free=free, obstacle=obstacle,
                        traversable=traversable)
    meta = {"origin": [float(xmin), float(ymin)], "res": RES, "floor_z": floor_z,
            "shape": [H, W], "scan": os.path.abspath(scan), "up": up}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    if not os.path.exists(os.path.join(out, "places.json")):
        json.dump({}, open(os.path.join(out, "places.json"), "w"))
    render_map(out)
    print(f"floor at z={floor_z:.2f} m, grid {W}x{H} @ {RES} m, "
          f"{free.sum() * RES * RES:.1f} m² walkable -> {out}/map.png")


# ---------------- map I/O ----------------

class Map:
    def __init__(self, d: str):
        self.dir = d
        g = np.load(os.path.join(d, "grid.npz"))
        self.free, self.obstacle, self.trav = g["free"], g["obstacle"], g["traversable"]
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.ox, self.oy = self.meta["origin"]
        self.places = json.load(open(os.path.join(d, "places.json")))
        self.hit = _dilate(self.obstacle | ~self.free, int(round(BODY_RADIUS / RES)))

    def cell(self, x, y):
        return int((y - self.oy) / RES), int((x - self.ox) / RES)

    def xy(self, r, c):
        return self.ox + (c + 0.5) * RES, self.oy + (r + 0.5) * RES

    def inside(self, r, c):
        return 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1]

    def place(self, name_or_xy: str):
        if name_or_xy in self.places:
            return tuple(self.places[name_or_xy]["xy"])
        try:
            x, y = (float(v) for v in name_or_xy.split(","))
            return x, y
        except ValueError:
            sys.exit(f"unknown place {name_or_xy!r}; known: {', '.join(self.places) or 'none'}")

    def start_pose(self):
        if "start" not in self.places:
            sys.exit("set the robot's start first: scan_map.py place <map> start X Y --yaw DEG")
        s = self.places["start"]
        return s["xy"][0], s["xy"][1], math.radians(s.get("yaw_deg", 0.0))

    # map <-> sim (sim origin = start pose, facing +x)
    def to_sim(self, x, y):
        sx, sy, th = self.start_pose()
        dx, dy = x - sx, y - sy
        c, s = math.cos(-th), math.sin(-th)
        return c * dx - s * dy, s * dx + c * dy

    def to_map(self, x, y, yaw=0.0):
        sx, sy, th = self.start_pose()
        c, s = math.cos(th), math.sin(th)
        return sx + c * x - s * y, sy + s * x + c * y, yaw + th


def render_map(d: str, path=None, traj=None, out_name="map.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = Map(d)
    H, W = m.free.shape
    img = np.full((H, W, 3), 0.55)               # unknown: grey
    img[m.free] = (0.96, 0.96, 0.94)             # walkable floor
    img[m.free & ~m.trav] = (0.85, 0.88, 0.95)   # too close to walls for the K1
    img[m.obstacle] = (0.2, 0.2, 0.25)           # walls / furniture
    ext = [m.ox, m.ox + W * RES, m.oy, m.oy + H * RES]
    fig, ax = plt.subplots(figsize=(max(6, W * RES * 0.9), max(5, H * RES * 0.9)))
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.set_xticks(np.arange(math.floor(ext[0]), ext[1] + 1, 1.0))
    ax.set_yticks(np.arange(math.floor(ext[2]), ext[3] + 1, 1.0))
    ax.grid(True, color="#4a90d9", alpha=0.25, lw=0.5)
    for name, p in m.places.items():
        ax.plot(*p["xy"], "o", color="#d0453a" if name == "start" else "#2a7d4f")
        ax.annotate(name, p["xy"], xytext=(4, 4), textcoords="offset points", fontsize=9)
        if "yaw_deg" in p:
            a = math.radians(p["yaw_deg"])
            ax.arrow(*p["xy"], 0.4 * math.cos(a), 0.4 * math.sin(a), width=0.03, color="#d0453a")
    if path:
        ax.plot(*zip(*path), "-o", color="#e08a1e", lw=2, ms=4, label="plan")
    if traj:
        ax.plot(*zip(*traj), "-", color="#7b3fb5", lw=1.5, label="K1 in twin")
    if path or traj:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(d, out_name), dpi=110)
    plt.close(fig)


# ---------------- planning ----------------

def _nearest_trav(m: Map, r, c, max_cells=12):
    if m.inside(r, c) and m.trav[r, c]:
        return r, c
    for rad in range(1, max_cells + 1):
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad) if abs(dr) != rad else range(-rad, rad + 1):
                if m.inside(r + dr, c + dc) and m.trav[r + dr, c + dc]:
                    return r + dr, c + dc
    return None


def _line_clear(m: Map, a, b, mask=None):
    mask = m.trav if mask is None else mask
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for t in np.linspace(0, 1, n + 1):
        r, c = round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)
        if not mask[r, c]:
            return False
    return True


def plan(m: Map, goal: str, start_xy=None, los_mask=None, clearance=None, prefer=0.8, weight=3.0):
    """A* on the traversable grid; shortcuts are only taken where `los_mask` (default: traversable) is clear.
    With `clearance` (metres to the nearest obstacle per cell), cells closer than `prefer` cost extra, so
    routes keep to the middle of corridors instead of hugging walls and furniture."""
    sx, sy = start_xy if start_xy is not None else m.start_pose()[:2]
    gx, gy = m.place(goal)
    s = _nearest_trav(m, *m.cell(sx, sy))
    g = _nearest_trav(m, *m.cell(gx, gy))
    if s is None or g is None:
        return None, f"{'start' if s is None else goal} is not reachable (inside or next to a wall)"
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    openq = [(0.0, s)]
    cost, prev = {s: 0.0}, {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for dr, dc, w in nbrs:
            nb = (cur[0] + dr, cur[1] + dc)
            if not m.inside(*nb) or not m.trav[nb]:
                continue
            nc = cost[cur] + w * (1.0 if clearance is None else
                                  1.0 + weight * max(0.0, prefer - float(clearance[nb])) / prefer)
            if nc < cost.get(nb, 1e18):
                cost[nb], prev[nb] = nc, cur
                heapq.heappush(openq, (nc + math.dist(nb, g), nb))
    if g not in cost:
        return None, f"no walkable path from start to {goal}"
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells.reverse()
    keep = [cells[0]]  # line-of-sight shortcutting -> few waypoints
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_clear(m, cells[i], cells[j], los_mask):
            j -= 1
        if j == i + 1 and los_mask is not None:  # narrow passage: shortcut with the normal clearance
            j = len(cells) - 1
            while j > i + 1 and not _line_clear(m, cells[i], cells[j]):
                j -= 1
        keep.append(cells[j])
        i = j
    pts = [m.xy(*c) for c in keep]
    return pts, f"{len(pts) - 1} legs, {sum(math.dist(a, b) for a, b in zip(pts, pts[1:])):.1f} m"


def sim_walls(m: Map, cell=0.10, height=1.2):
    """Merge obstacle cells (downsampled to `cell`) into boxes in the sim frame."""
    k = max(1, int(round(cell / RES)))
    H, W = m.obstacle.shape
    occ = m.obstacle[:H - H % k, :W - W % k].reshape(H // k, k, W // k, k).any(axis=(1, 3))
    used = np.zeros_like(occ)
    _, _, th = m.start_pose()
    boxes = []
    for r in range(occ.shape[0]):
        c = 0
        while c < occ.shape[1]:
            if not occ[r, c] or used[r, c]:
                c += 1
                continue
            c2 = c
            while c2 + 1 < occ.shape[1] and occ[r, c2 + 1] and not used[r, c2 + 1]:
                c2 += 1
            r2 = r
            while r2 + 1 < occ.shape[0] and occ[r2 + 1, c:c2 + 1].all() and not used[r2 + 1, c:c2 + 1].any():
                r2 += 1
            used[r:r2 + 1, c:c2 + 1] = True
            cx = m.ox + (c + c2 + 1) / 2 * cell
            cy = m.oy + (r + r2 + 1) / 2 * cell
            sx, sy = m.to_sim(cx, cy)
            boxes.append((sx, sy, (c2 - c + 1) * cell / 2, (r2 - r + 1) * cell / 2, height, -th))
            c = c2 + 1
    return boxes


# ---------------- rehearsal in the MuJoCo twin ----------------

def rehearse(m: Map, path, gif: bool = False, max_v=0.4, max_w=0.8):
    from k1_sim import K1Sim

    sim = K1Sim(walls=sim_walls(m), with_props=False)
    sim.ctrl.update_state()
    sim.ctrl.start()
    dt = sim.dt
    for _ in range(int(1.5 / dt)):  # let it settle standing
        sim.step()
    wps = [m.to_sim(*p) for p in path[1:]]
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    t_max = 20 + 3 * length / 0.3
    traj, frames = [], []
    i, t, result = 0, 0.0, None
    while result is None:
        x, y, yaw, _ = sim.pose()
        mx, my, _ = m.to_map(x, y)
        traj.append((mx, my))
        r, c = m.cell(mx, my)
        if sim.fallen:
            result = ("fell", False)
        elif not m.inside(r, c) or m.hit[r, c]:
            result = (f"hit an obstacle at map ({mx:.2f}, {my:.2f})", False)
        elif t > t_max:
            result = ("timed out", False)
        else:
            tx, ty = wps[i]
            d = math.hypot(tx - x, ty - y)
            last = i == len(wps) - 1
            if d < (0.35 if last else 0.25):
                if last:
                    result = ("arrived", True)
                    sim.set_velocity(0, 0, 0)
                else:
                    i += 1
                continue
            err = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.35:  # turn in place first
                sim.set_velocity(0, 0, float(np.clip(1.5 * err, -max_w, max_w)))
            else:
                v = float(np.clip(d, 0.2, max_v))  # policy ignores < ~0.2 m/s
                sim.set_velocity(v, 0, float(np.clip(1.2 * err, -max_w, max_w)))
        if gif and int(t / dt) % int(0.25 / dt) == 0:
            frames.append(sim.render("chase"))
        sim.step()
        t += dt
    for _ in range(int(0.5 / dt)):
        sim.step()
    reason, ok = result
    out = {"ok": ok, "result": reason, "sim_time_s": round(t, 1), "path_m": round(length, 2),
           "waypoints_reached": i + (1 if ok else 0), "waypoints": len(wps)}
    render_map(m.dir, path=path, traj=traj, out_name="rehearsal.png")
    if gif and frames:
        from PIL import Image

        ims = [Image.fromarray(f).resize((480, 360)) for f in frames]
        ims[0].save(os.path.join(m.dir, "rehearsal.gif"), save_all=True,
                    append_images=ims[1:], duration=125, loop=0)
    return out


# ---------------- synthetic test scan ----------------

def synth(out_path: str):
    """A Y-up 'iPhone-style' scan: hallway + living room + kitchen with a counter."""
    import trimesh

    parts = []

    def box(cx, cy, sx, sy, h, z0=0.0):
        b = trimesh.creation.box(extents=[sx, sy, h])
        b.apply_translation([cx, cy, z0 + h / 2])
        parts.append(b)

    box(4, 3, 8.2, 6.2, 0.02, -0.02)                                      # floor
    box(4, -0.05, 8.2, 0.1, 2.5); box(4, 6.05, 8.2, 0.1, 2.5)             # outer walls
    box(-0.05, 3, 0.1, 6.2, 2.5); box(8.05, 3, 0.1, 6.2, 2.5)
    box(4.5, 1.1, 0.1, 2.2, 2.5); box(4.5, 4.9, 0.1, 2.2, 2.5)            # inner wall, 1.6 m door
    box(7.6, 4.5, 0.6, 2.5, 0.9)                                          # kitchen counter
    box(2.0, 5.3, 2.0, 0.9, 0.8)                                          # couch
    box(2.5, 1.5, 0.8, 0.8, 0.75)                                         # table
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))  # Z-up -> Y-up
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    mesh.export(out_path)
    print(f"wrote {out_path}  (try: place start 0.8 3.0 --yaw 0, place kitchen 6.8 4.5)")


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("scan"); p.add_argument("map_dir")
    p.add_argument("--up", choices=["y", "z"], default="y", help="iPhone exports are Y-up")
    p = sub.add_parser("place"); p.add_argument("map_dir"); p.add_argument("name")
    p.add_argument("x", type=float); p.add_argument("y", type=float); p.add_argument("--yaw", type=float)
    p = sub.add_parser("plan"); p.add_argument("map_dir"); p.add_argument("goal")
    p = sub.add_parser("rehearse"); p.add_argument("map_dir"); p.add_argument("goal")
    p.add_argument("--gif", action="store_true")
    p = sub.add_parser("synth"); p.add_argument("out")
    a = ap.parse_args()

    if a.cmd == "build":
        build(a.scan, a.map_dir, a.up)
    elif a.cmd == "synth":
        synth(a.out)
    elif a.cmd == "place":
        f = os.path.join(a.map_dir, "places.json")
        places = json.load(open(f))
        places[a.name] = {"xy": [a.x, a.y], **({"yaw_deg": a.yaw} if a.yaw is not None else {})}
        json.dump(places, open(f, "w"), indent=2)
        render_map(a.map_dir)
        print(f"{a.name} = ({a.x}, {a.y}) -> {a.map_dir}/map.png")
    else:
        m = Map(a.map_dir)
        path, msg = plan(m, a.goal)
        if path is None:
            sys.exit(f"plan failed: {msg}")
        json.dump({"goal": a.goal, "map": [list(p) for p in path],
                   "sim": [list(m.to_sim(*p)) for p in path]},
                  open(os.path.join(a.map_dir, f"route_{a.goal}.json"), "w"), indent=2)
        render_map(a.map_dir, path=path, out_name=f"route_{a.goal}.png")
        print(f"route to {a.goal}: {msg} -> {a.map_dir}/route_{a.goal}.png")
        if a.cmd == "rehearse":
            r = rehearse(m, path, gif=a.gif)
            print(json.dumps(r))
            print(f"-> {a.map_dir}/rehearsal.png" + (f", {a.map_dir}/rehearsal.gif" if a.gif else ""))
            sys.exit(0 if r["ok"] else 2)


if __name__ == "__main__":
    main()
