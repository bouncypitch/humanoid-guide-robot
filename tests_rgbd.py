"""End-to-end check of rgbd_ingest.py + map_clean.py on the synthetic session.

    .venv/bin/python tests_rgbd.py                 # folder + zip + live, no Gemma (fast)
    .venv/bin/python tests_rgbd.py --label         # also run Gemma labelling (~20 s/frame)
    .venv/bin/python tests_rgbd.py --only verify --map maps/session_fake --session captures/session_fake

Checks in the final grid (maps/<name>): bag cells and every person position are free of
obstacles, walls and the counter are still obstacles, the doorway is traversable and
scan_map.plan finds a route living room -> kitchen.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable
import scan_map  # noqa: E402
from rgbd_ingest import W_YUP_TO_ZUP  # noqa: E402


def run(*args):
    print("$", " ".join(args), flush=True)
    t = time.time()
    subprocess.run([PY, *args], check=True, cwd=HERE)
    print(f"  ({time.time() - t:.1f}s)")


def room_to_map(session, map_dir):
    gt = json.load(open(os.path.join(session, "gt.json")))
    T_level = np.array(json.load(open(os.path.join(map_dir, "clean_report.json")))["level_transform"])
    T = T_level @ W_YUP_TO_ZUP @ np.array(gt["room_to_arkit"])
    return gt, lambda x, y, z=0.0: (T @ [x, y, z, 1.0])[:3]


def cells_in_disc(m, c, r):
    out = []
    for dx in np.arange(-r, r + 1e-9, scan_map.RES / 2):
        for dy in np.arange(-r, r + 1e-9, scan_map.RES / 2):
            if dx * dx + dy * dy <= r * r:
                rr, cc = m.cell(c[0] + dx, c[1] + dy)
                if m.inside(rr, cc):
                    out.append((rr, cc))
    return sorted(set(out))


def frac(grid, cells, tol=0):
    from scipy import ndimage

    g = ndimage.binary_dilation(grid, iterations=tol) if tol else grid
    return float(np.mean([g[c] for c in cells])) if cells else float("nan")


def verify(session, map_dir, label=True):
    gt, to_map = room_to_map(session, map_dir)
    m = scan_map.Map(map_dir)
    before = scan_map.Map(os.path.join(map_dir, "_before"))
    res, ok = {}, True

    def check(name, cond, val):
        nonlocal ok
        res[name] = {"value": val, "pass": bool(cond)}
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {val}")

    # bag: its footprint (+5 cm) must not be obstacle in the cleaned map
    (bx, by, _), (sx, sy, _) = gt["bag"]["center"], gt["bag"]["size"]
    bc = to_map(bx, by)
    bag_cells = cells_in_disc(m, bc[:2], max(sx, sy) / 2)
    bag_before = frac(before.obstacle, cells_in_disc(before, bc[:2], max(sx, sy) / 2))
    bag_after = frac(m.obstacle, bag_cells)
    if label:
        check("bag removed (obstacle frac in footprint, before -> after)", bag_after < 0.05,
              f"{bag_before:.2f} -> {bag_after:.2f}")
    else:
        res["bag_no_label"] = f"{bag_before:.2f} -> {bag_after:.2f} (expected to stay without --label)"
        print(f"  [info] bag without labels: {bag_before:.2f} -> {bag_after:.2f}")
    # person: no obstacle within 0.3 m of any of its positions (ignore spots next to walls/furniture)
    worst = 0.0
    for px, py in gt["person_track"]:
        c = to_map(px, py)
        cells = cells_in_disc(m, c[:2], 0.3)
        near_static = frac(before.obstacle, cells) > 0 and min(px, 8 - px, py, 6 - py, abs(px - 4.5)) < 0.7
        if not near_static:
            worst = max(worst, frac(m.obstacle, cells))
    check("person gone (max obstacle frac over 60 positions)", worst < 0.05, round(worst, 3))
    # walls: sample every 10 cm along every wall face, 1-cell tolerance
    pts = [(x, 0.0) for x in np.arange(0.3, 7.8, 0.1)] + [(x, 6.0) for x in np.arange(0.3, 7.8, 0.1)]
    pts += [(0.0, y) for y in np.arange(0.3, 5.8, 0.1)] + [(8.0, y) for y in np.arange(0.3, 5.8, 0.1)]
    pts += [(4.5, y) for y in np.arange(0.2, 2.1, 0.1)] + [(4.5, y) for y in np.arange(3.9, 5.9, 0.1)]
    wall_cells = [m.cell(*to_map(x, y)[:2]) for x, y in pts]
    wall_cells = [c for c in wall_cells if m.inside(*c)]
    check("walls remain (obstacle frac along walls, 1-cell tol)", frac(m.obstacle, wall_cells, 1) > 0.95,
          round(frac(m.obstacle, wall_cells, 1), 3))
    # counter footprint
    (cx, cy, _), (csx, csy, _) = gt["counter"]["center"], gt["counter"]["size"]
    cc = [m.cell(*to_map(x, y)[:2]) for x in np.arange(cx - csx / 2 + 0.1, cx + csx / 2 - 0.05, 0.1)
          for y in np.arange(cy - csy / 2 + 0.1, cy + csy / 2 - 0.05, 0.1)]
    x0, x1, y0, y1 = cx - csx / 2, cx + csx / 2, cy - csy / 2, cy + csy / 2
    edge = [(x, y0) for x in np.arange(x0, x1, 0.05)] + [(x, y1) for x in np.arange(x0, x1, 0.05)]
    edge += [(x0, y) for y in np.arange(y0, y1, 0.05)]  # back face (x1) is 10 cm from the wall: never seen
    ec = [m.cell(*to_map(x, y)[:2]) for x, y in edge]
    cb = [before.cell(*to_map(x, y)[:2]) for x in np.arange(x0 + 0.1, x1 - 0.05, 0.1)
          for y in np.arange(y0 + 0.1, y1 - 0.05, 0.1)]
    f_e, f_in, f_b = frac(m.obstacle, ec, 1), frac(m.obstacle, cc), frac(before.obstacle, cb)
    f_free = frac(m.free, cc)  # interior only: boundary cells may hold the visible floor gap at the wall
    check("counter remains (no free cell in footprint; visible outline obstacle; footprint obstacle before -> after)",
          f_free == 0 and f_e > 0.8 and f_in >= f_b - 0.05,
          f"free {f_free:.2f}; outline {f_e:.2f}; footprint {f_b:.2f} -> {f_in:.2f} (rest = unseen, not free)")
    # doorway: centre traversable + A* route living room -> kitchen through it
    door = to_map(4.5, 3.0)
    check("doorway centre traversable", bool(m.trav[m.cell(*door[:2])]), f"cell {m.cell(*door[:2])}")
    a, b = to_map(*gt["living_room_point"]), to_map(*gt["kitchen_point"])
    places = {"start": {"xy": [float(a[0]), float(a[1])], "yaw_deg": 0.0},
              "kitchen": {"xy": [float(b[0]), float(b[1])]}, "door": {"xy": [float(door[0]), float(door[1])]}}
    json.dump(places, open(os.path.join(map_dir, "places.json"), "w"), indent=2)
    m = scan_map.Map(map_dir)
    path, msg = scan_map.plan(m, "kitchen")
    passes_door = path is not None and any(abs(math.dist(p, door[:2])) < 0.9 for p in
                                          [tuple(np.array(p0) + t * (np.array(p1) - np.array(p0)))
                                           for p0, p1 in zip(path, path[1:]) for t in np.linspace(0, 1, 20)])
    check("A* route living room -> kitchen via doorway", path is not None and passes_door, msg)
    if path:
        scan_map.render_map(map_dir, path=path, out_name="route_kitchen.png")
    scan_map.render_map(map_dir)
    json.dump({"ok": ok, **res}, open(os.path.join(map_dir, "verify.json"), "w"), indent=2)
    return ok


async def stream_session(session, uri="ws://127.0.0.1:8770/ingest", skip=()):
    """Tiny client: stream a session folder like the phone app does."""
    from websockets.asyncio.client import connect

    async with connect(uri, max_size=64 * 2 ** 20) as ws:
        await ws.send(open(os.path.join(session, "session.json")).read())
        for jf in sorted(glob.glob(os.path.join(session, "frames", "[0-9]*.json"))):
            stem = jf[:-5]
            meta = json.load(open(jf))
            if meta["index"] in skip:
                continue
            files = [p for p in (stem + ".jpg", stem + "_depth.png", stem + "_conf.png", stem + "_mask.png")
                     if os.path.exists(p)]
            meta["files"] = [os.path.basename(p) for p in files]
            await ws.send(json.dumps(meta))
            for p in files:
                await ws.send(open(p, "rb").read())
        await ws.send(json.dumps({"end": True}))
        reply = json.loads(await asyncio.wait_for(ws.recv(), 600))
    return reply


async def live_test(session, captures, scans, label):
    import rgbd_ingest

    stop = asyncio.Event()
    srv = asyncio.create_task(rgbd_ingest.serve("127.0.0.1", 8770, captures, scans, 10, label, 3, stop))
    await asyncio.sleep(1.0)
    try:
        reply = await stream_session(session, skip={7, 8, 40})  # the app may skip keyframes
    finally:
        stop.set()
        await srv
    return reply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--only", choices=["verify"])
    ap.add_argument("--session", default="captures/session_fake")
    ap.add_argument("--map", default="maps/session_fake")
    ap.add_argument("--skip-live", action="store_true")
    a = ap.parse_args()
    os.chdir(HERE)
    if a.only == "verify":
        sys.exit(0 if verify(a.session, a.map, a.label) else 1)
    lab = ["--label", "--label-every", "3"] if a.label else []
    results = {}
    # 1. folder path
    run("tools/fake_session.py", "captures/session_fake", "--zip")
    run("rgbd_ingest.py", "captures/session_fake", "scans/session_fake", *lab)
    run("map_clean.py", "scans/session_fake", "maps/session_fake")
    print("== verify folder path")
    results["folder"] = verify("captures/session_fake", "maps/session_fake", a.label)
    # 2. zip path (reuse the folder run's objects.json so Gemma isn't run twice)
    run("rgbd_ingest.py", "captures/session_fake.zip", "scans/session_fake_zip")
    if a.label:
        shutil.copy("scans/session_fake/objects.json", "scans/session_fake_zip/objects.json")
    run("map_clean.py", "scans/session_fake_zip", "maps/session_fake_zip")
    print("== verify zip path")
    results["zip"] = verify("captures/session_fake", "maps/session_fake_zip", a.label)
    # 3. live WebSocket path
    if not a.skip_live:
        print("== live WebSocket ingest")
        reply = asyncio.run(live_test("captures/session_fake", "captures/live_test", "scans/live_test", False))
        print("  server reply:", reply)
        out = reply["out"]
        if a.label:
            shutil.copy("scans/session_fake/objects.json", os.path.join(out, "objects.json"))
        assert os.path.exists(os.path.join(out, "preview.png")), "no live preview.png"
        live_map = "maps/session_fake_live"
        run("map_clean.py", out, live_map)
        shutil.copy("captures/session_fake/gt.json", os.path.join(reply["session"], "gt.json"))  # same poses
        print("== verify live path")
        results["live"] = verify(reply["session"], live_map, a.label)
        results["live_fused"] = reply["fused"]
    print(json.dumps(results))
    sys.exit(0 if all(v for k, v in results.items() if k in ("folder", "zip", "live")) else 1)


if __name__ == "__main__":
    main()
