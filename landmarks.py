"""Find named landmarks in an RGB-D walkthrough with Gemma 4 and put them on the map.

    python landmarks.py captures/session_20261024-1347.zip maps/agihouse --every 6
    python landmarks.py ... --limit 10        # quick test on the first 10 sampled frames
    python landmarks.py --replot maps/agihouse  # redo the map overlay from saved detections

For every Nth keyframe Gemma 4 (Ollama, on-device) returns boxes for the house's landmarks
(front entrance, restroom, bar, dining table, TV, cooktop, kitchen, trash can, poker table,
couch...). Each box is lifted to 3D with the frame's depth + pose (rgbd_ingest.lift_box), moved
into the levelled map frame (map_clean's level_transform), and repeated sightings are
clustered. Each landmark gets:
  places.json  "<name>": {"xy": approach point the robot walks to, "object_xy", "kind": "landmark",
                          "desc", "sightings", "first_seen_frame", "hero", "sim": {...}}
  landmarks/<name>.jpg   best crop (for slides / the dashboard / Gemma's localization prompts)
  landmarks.png          floor plan with every landmark pinned
Detections are cached in landmarks_raw.jsonl, so reruns and --replot are cheap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rgbd_ingest as ri  # noqa: E402
import scan_map  # noqa: E402

MODEL = "gemma4:e4b-it-qat"

LANDMARKS = {
    "front_entrance": "the house's front door / main entrance",
    "restroom": "a restroom or bathroom door, or a toilet / bathroom sink",
    "bar": "a bar counter (stools, bottles, glasses)",
    "dining_table": "a dining table with chairs",
    "tv": "a television / large screen",
    "cooktop": "a stove or cooktop",
    "kitchen": "the kitchen area (counter, sink, cabinets, fridge)",
    "fridge": "a refrigerator",
    "trash_can": "a trash can or recycling bin",
    "poker_table": "a poker / card table (felt top)",
    "couch": "a couch or sofa",
    "stairs": "a staircase",
    "piano": "a piano",
    "fireplace": "a fireplace",
}
COLORS = {  # rgba stand-ins for the MuJoCo twin
    "front_entrance": [0.55, 0.35, 0.2, 1], "restroom": [0.3, 0.6, 0.95, 1], "bar": [0.6, 0.3, 0.15, 1],
    "dining_table": [0.75, 0.55, 0.35, 1], "tv": [0.1, 0.1, 0.12, 1], "cooktop": [0.2, 0.2, 0.2, 1],
    "kitchen": [0.9, 0.9, 0.88, 1], "fridge": [0.85, 0.87, 0.9, 1], "trash_can": [0.2, 0.5, 0.25, 1],
    "poker_table": [0.1, 0.45, 0.2, 1], "couch": [0.45, 0.45, 0.55, 1], "stairs": [0.6, 0.5, 0.4, 1],
    "piano": [0.05, 0.05, 0.05, 1], "fireplace": [0.55, 0.25, 0.2, 1],
}

PROMPT = ("You are mapping a house for a guide robot. Find these landmarks in the photo, only if clearly "
          "visible:\n" + "\n".join(f"- {k}: {v}" for k, v in LANDMARKS.items()) +
          "\nMost photos show zero to two landmarks. Only include a landmark you can clearly see; never "
          "guess from context, and never give two landmarks the same box. Return a tight bounding box "
          "box_2d = [ymin, xmin, ymax, xmax] normalised to 0-1000, and an honest confidence 0-1. "
          "Return an empty list if none are visible.")
SCHEMA = {"type": "object", "properties": {"landmarks": {"type": "array", "items": {
    "type": "object", "properties": {
        "name": {"type": "string", "enum": list(LANDMARKS)},
        "box_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "confidence": {"type": "number"}},
    "required": ["name", "box_2d", "confidence"]}}}, "required": ["landmarks"]}


def level_T(map_dir):
    rep = json.load(open(os.path.join(map_dir, "clean_report.json")))
    return np.array(rep["level_transform"], float)


def orientation(f) -> int:
    """Degrees to rotate the stored (sensor-landscape) image CCW so the world is upright:
    the direction of world-up in the camera image picks it (phone held portrait -> 270)."""
    R = np.asarray(f.meta["pose"], float)[:3, :3]
    up = R.T @ np.array([0.0, 1.0, 0.0])  # world up in GL camera coords (+x right, +y up)
    if abs(up[0]) > abs(up[1]):
        return 90 if up[0] > 0 else 270
    return 0 if up[1] > 0 else 180


def upright_path(f, rot: int) -> str:
    from PIL import Image

    if rot == 0:
        return f.rgb_path
    d = os.path.join(os.path.dirname(os.path.dirname(f.rgb_path)), "upright")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, os.path.basename(f.rgb_path))
    if not os.path.exists(p):
        Image.open(f.rgb_path).rotate(rot, expand=True).save(p, quality=92)
    return p


def unrotate_box(b, rot: int):
    """[ymin,xmin,ymax,xmax] (0-1000) on the upright image -> same on the stored image."""
    y0, x0, y1, x1 = (float(v) for v in b[:4])
    if rot == 270:   # stored x = upright y ; stored y = 1000 - upright x
        return [1000 - x1, y0, 1000 - x0, y1]
    if rot == 90:    # stored x = 1000 - upright y ; stored y = upright x
        return [x0, 1000 - y1, x1, 1000 - y0]
    if rot == 180:
        return [1000 - y1, 1000 - x1, 1000 - y0, 1000 - x0]
    return [y0, x0, y1, x1]


def detect(frames, every, limit, cache, model, host=None):
    import ollama

    client = ollama.Client(host=host, timeout=180)
    done = set()
    if os.path.exists(cache):
        for ln in open(cache):
            done.add(json.loads(ln)["frame"])
    sampled = [f for i, f in enumerate(frames) if i % every == 0]
    if limit:
        sampled = sampled[:limit]
    todo = [f for f in sampled if f.index not in done]
    print(f"{len(sampled)} frames sampled, {len(done)} cached, {len(todo)} to label with {model}")
    t0 = time.time()
    with open(cache, "a") as out:
        for k, f in enumerate(todo):
            rot = orientation(f)
            try:
                r = client.chat(model=model, format=SCHEMA, options={"temperature": 0},
                                messages=[{"role": "user", "content": PROMPT, "images": [upright_path(f, rot)]}])
                dets = json.loads(r["message"]["content"]).get("landmarks", [])
                for d in dets:
                    if isinstance(d.get("box_2d"), list) and len(d["box_2d"]) == 4:
                        d["box_upright"] = d["box_2d"]
                        d["box_2d"] = unrotate_box(d["box_2d"], rot)
            except Exception as e:  # noqa: BLE001
                print(f"  frame {f.index}: {e}", file=sys.stderr)
                dets = []
            out.write(json.dumps({"frame": f.index, "dets": dets}) + "\n")
            out.flush()
            if k % 10 == 0 or dets:
                el = time.time() - t0
                eta = el / (k + 1) * (len(todo) - k - 1)
                names = ", ".join(f"{d.get('name')}({d.get('confidence', 0):.2f})" for d in dets) or "-"
                print(f"  [{k + 1}/{len(todo)} {el:.0f}s eta {eta:.0f}s] frame {f.index}: {names}", flush=True)


def lift_all(frames_by_idx, cache, L):
    obs = []
    for ln in open(cache):
        rec = json.loads(ln)
        f = frames_by_idx.get(rec["frame"])
        if f is None:
            continue
        h, w = f.rgb.shape[:2]
        for d in rec["dets"]:
            if d.get("name") not in LANDMARKS or float(d.get("confidence", 0)) < 0.35:
                continue
            b = ri.normalise_box(d.get("box_2d"), w, h)
            if b is None or (b[2] - b[0]) * (b[3] - b[1]) > 0.9:  # whole-frame boxes are useless
                continue
            lifted = ri.lift_box(f, b)
            if lifted is None:
                continue
            c, ext, n = lifted
            cm = (L @ np.append(c, 1.0))[:3]
            cam = (L @ f.T[:, 3])[:3]
            obs.append({"name": d["name"], "frame": f.index, "conf": float(d["confidence"]), "box": b,
                        "xyz": cm.tolist(), "ext": ext.tolist(), "cam": cam.tolist(),
                        "area": (b[2] - b[0]) * (b[3] - b[1]), "rgb_path": f.rgb_path})
    return obs


# Gemma over-labels (every door is a "front_entrance"), so keep only the best-supported clusters per kind.
MAX_PER_KIND = {"front_entrance": 1, "kitchen": 1, "cooktop": 1, "fridge": 1, "bar": 1, "piano": 1, "fireplace": 1,
                "poker_table": 1, "stairs": 2, "restroom": 2, "dining_table": 2, "couch": 3, "tv": 3, "trash_can": 3}
AREAS = {"kitchen"}  # whole areas: a tight crop (e.g. of the sink) can't be judged "the kitchen", so not verified
MAX_RANGE = 4.5  # m; LiDAR depth (and Gemma's boxes) are unreliable beyond this


def crop_upright(p, pad=0.08):
    """The sighting's box cropped from its frame, padded and rotated upright."""
    from PIL import Image

    im = Image.open(p["rgb_path"]).convert("RGB")
    rot = orientation(ri.load_frame(os.path.dirname(p["rgb_path"]), os.path.basename(p["rgb_path"])[:-4]))
    W, H = im.size
    y0, x0, y1, x1 = p["box"]
    crop = im.crop((int(max(0, x0 - pad) * W), int(max(0, y0 - pad) * H),
                    int(min(1, x1 + pad) * W), int(min(1, y1 + pad) * H)))
    return crop.rotate(rot, expand=True)


class Verifier:
    """Second Gemma pass on a cropped sighting: "is this really a <landmark>?" (cached)."""

    def __init__(self, cache, model=MODEL):
        import ollama

        self.client, self.model, self.path = ollama.Client(timeout=180), model, cache
        self.cache = json.load(open(cache)) if os.path.exists(cache) else {}

    def __call__(self, name, p) -> bool:
        key = f"{p['frame']}:{name}:{','.join(f'{v:.3f}' for v in p['box'])}"
        if key not in self.cache:
            path = os.path.join(os.path.dirname(self.path), ".verify.jpg")
            c = crop_upright(p)
            c.thumbnail((768, 768))
            c.save(path, quality=90)
            r = self.client.chat(model=self.model, options={"temperature": 0}, format={
                "type": "object", "properties": {"answer": {"type": "boolean"}}, "required": ["answer"]},
                messages=[{"role": "user", "images": [path], "content":
                           f"Is the main subject of this photo {LANDMARKS[name]}? Be strict: a different "
                           "object that merely looks similar (whiteboard, poster, floor pattern, bag, chair) "
                           "is not. Answer true or false."}])
            self.cache[key] = bool(json.loads(r["message"]["content"])["answer"])
            json.dump(self.cache, open(self.path, "w"), indent=0)
        return self.cache[key]


def cluster(obs, eps=2.0, min_sightings=3, verify=None):
    from sklearn.cluster import DBSCAN

    obs = [o for o in obs if np.hypot(o["xyz"][0] - o["cam"][0], o["xyz"][1] - o["cam"][1]) <= MAX_RANGE]
    out = []
    for name in sorted({o["name"] for o in obs}):
        pts = [o for o in obs if o["name"] == name]
        X = np.array([o["xyz"][:2] for o in pts])
        lab = DBSCAN(eps=eps, min_samples=1).fit(X).labels_
        groups = sorted(({i: [p for p, l in zip(pts, lab) if l == i] for i in set(lab)}).values(),
                        key=lambda g: -(len(g) + 0.1 * sum(p["conf"] for p in g)))
        groups = [g for g in groups if len(g) >= min_sightings]
        if verify and name not in AREAS:  # keep a cluster only if Gemma confirms most of its (up to 3 biggest) crops
            kept = []
            for g in groups:
                if len(kept) == MAX_PER_KIND.get(name, 2):
                    break
                top = sorted(g, key=lambda p: -p["area"])[:3]
                yes = [p for p in top if verify(name, p)]
                print(f"  verify {name}: {len(yes)}/{len(top)} confirmed" + ("" if 2 * len(yes) > len(top) else "  -> dropped"))
                if 2 * len(yes) > len(top):
                    kept.append((g, max(yes, key=lambda p: p["conf"] * min(p["area"], 0.35))))
        else:
            kept = [(g, max(g, key=lambda p: p["conf"] * min(p["area"], 0.35)))
                    for g in groups[:MAX_PER_KIND.get(name, 2)]]
        for k, (g, hero) in enumerate(kept):
            xyz = np.median([p["xyz"] for p in g], axis=0)
            ext = np.median([p["ext"] for p in g], axis=0)
            out.append({"name": name if k == 0 else f"{name}_{k + 1}", "kind": name, "xyz": xyz, "ext": ext,
                        "cams": np.array([p["cam"][:2] for p in g]), "sightings": len(g),
                        "conf": float(np.mean([p["conf"] for p in g])),
                        "first_seen_frame": min(p["frame"] for p in g), "hero": hero})
    return out


def reachable(m: scan_map.Map):
    """Traversable cells connected to the robot's start (all traversable cells if no start is set)."""
    from scipy import ndimage

    if "start" not in m.places:
        return m.trav
    lab, _ = ndimage.label(m.trav)
    r, c = m.cell(*m.places["start"]["xy"])
    near = lab[max(0, r - 10):r + 11, max(0, c - 10):c + 11]
    ids = near[near > 0]
    return lab == np.bincount(ids).argmax() if ids.size else m.trav


def approach_point(m: scan_map.Map, obj_xy, cams, standoff=0.8, reach=None):
    """Nearest reachable cell to the object, preferring the side the camera saw it from."""
    toward = np.median(cams, axis=0) - np.asarray(obj_xy)
    toward = toward / (np.linalg.norm(toward) + 1e-6)
    best, best_score = None, 1e18
    r0, c0 = m.cell(*obj_xy)
    R = int(3.0 / scan_map.RES)
    rr, cc = np.mgrid[max(0, r0 - R):min(m.trav.shape[0], r0 + R), max(0, c0 - R):min(m.trav.shape[1], c0 + R)]
    ok = (m.trav if reach is None else reach)[rr, cc]
    if not ok.any():
        return None
    xs = m.ox + (cc[ok] + 0.5) * scan_map.RES
    ys = m.oy + (rr[ok] + 0.5) * scan_map.RES
    dx, dy = xs - obj_xy[0], ys - obj_xy[1]
    dist = np.hypot(dx, dy)
    side = (dx * toward[0] + dy * toward[1]) / (dist + 1e-6)
    score = np.abs(dist - standoff) + 0.6 * (1 - side)
    i = int(np.argmin(score))
    best, best_score = (float(xs[i]), float(ys[i])), score[i]
    return best


def write_places(map_dir, lms):
    from PIL import Image

    m = scan_map.Map(map_dir)
    places = json.load(open(os.path.join(map_dir, "places.json")))
    places = {k: v for k, v in places.items() if v.get("kind") != "landmark"}  # replace old landmarks
    os.makedirs(os.path.join(map_dir, "landmarks"), exist_ok=True)
    rows = []
    reach = reachable(m)
    rej_path = os.path.join(map_dir, "landmarks_reject.txt")  # human review: one landmark name per line
    rejected = {ln.split("#")[0].strip() for ln in open(rej_path)} - {""} if os.path.exists(rej_path) else set()
    for lm in lms:
        if lm["name"] in rejected:
            print(f"  - {lm['name']}: rejected in landmarks_reject.txt")
            continue
        xy = approach_point(m, lm["xyz"][:2], lm["cams"], reach=reach)
        if xy is None:
            print(f"  ! {lm['name']}: no reachable approach point within 3 m, skipped")
            continue
        crop = crop_upright(lm["hero"])
        crop.thumbnail((640, 640))
        hero_rel = f"landmarks/{lm['name']}.jpg"
        crop.save(os.path.join(map_dir, hero_rel), quality=90)
        ext = np.clip(lm["ext"], 0.15, 3.0)
        places[lm["name"]] = {
            "xy": [round(xy[0], 2), round(xy[1], 2)], "kind": "landmark", "landmark": lm["kind"],
            "object_xy": [round(float(lm["xyz"][0]), 2), round(float(lm["xyz"][1]), 2)],
            "object_z": round(float(lm["xyz"][2]), 2), "desc": LANDMARKS[lm["kind"]],
            "sightings": lm["sightings"], "confidence": round(lm["conf"], 2),
            "first_seen_frame": lm["first_seen_frame"], "hero": hero_rel,
            "sim": {"shape": "box", "size": [round(float(e) / 2, 2) for e in ext],
                    "pos": [round(float(v), 2) for v in lm["xyz"]], "rgba": COLORS.get(lm["kind"], [0.7, 0.7, 0.7, 1])}}
        rows.append((lm["name"], lm["sightings"], lm["conf"], xy))
    json.dump(places, open(os.path.join(map_dir, "places.json"), "w"), indent=2)
    plot(map_dir)
    print(f"{len(rows)} landmarks -> {map_dir}/places.json, {map_dir}/landmarks.png")
    for name, n, c, xy in rows:
        print(f"  {name:18s} sightings={n:3d} conf={c:.2f} approach=({xy[0]:.1f}, {xy[1]:.1f})")


def plot(map_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = scan_map.Map(map_dir)
    H, W = m.free.shape
    img = np.full((H, W, 3), 0.93)
    img[m.free] = (0.99, 0.985, 0.97)
    img[m.obstacle] = (0.22, 0.24, 0.3)
    ext = [m.ox, m.ox + W * scan_map.RES, m.oy, m.oy + H * scan_map.RES]
    fig, ax = plt.subplots(figsize=(12, 12 * H / W))
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    cmap = plt.get_cmap("tab20")
    for i, (name, p) in enumerate((k, v) for k, v in m.places.items() if v.get("kind") == "landmark"):
        col = cmap(i % 20)
        ox, oy = p["object_xy"]
        ax.plot([ox, p["xy"][0]], [oy, p["xy"][1]], "-", color=col, lw=1)
        ax.plot(ox, oy, "s", color=col, ms=9, mec="white")
        ax.plot(*p["xy"], "o", color=col, ms=5)
        ax.annotate(name.replace("_", " "), (ox, oy), xytext=(6, 6), textcoords="offset points", fontsize=10,
                    weight="bold", color=col, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(map_dir, "landmarks.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("map_dir")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min-sightings", type=int, default=3)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the second Gemma pass on crops")
    ap.add_argument("--recluster", action="store_true", help="re-cluster landmarks_obs.json only")
    a = ap.parse_args()
    cache = os.path.join(a.map_dir, "landmarks_raw.jsonl")
    if a.replot:
        plot(a.map_dir)
        return
    if a.recluster:  # reuse lifted sightings, no Gemma calls
        obs = json.load(open(os.path.join(a.map_dir, "landmarks_obs.json")))
        write_places(a.map_dir, cluster(obs, min_sightings=a.min_sightings, verify=_verifier(a)))
        return
    root = ri.unzip_session(a.session) if a.session.endswith(".zip") else ri.find_session_root(a.session)
    frames = list(ri.iter_session(root))
    detect(frames, a.every, a.limit, cache, a.model)
    obs = lift_all({f.index: f for f in frames}, cache, level_T(a.map_dir))
    json.dump(obs, open(os.path.join(a.map_dir, "landmarks_obs.json"), "w"))
    print(f"{len(obs)} landmark sightings lifted to 3D")
    write_places(a.map_dir, cluster(obs, min_sightings=a.min_sightings, verify=_verifier(a)))


def _verifier(a):
    return None if a.no_verify else Verifier(os.path.join(a.map_dir, "landmarks_verify.json"), a.model)


if __name__ == "__main__":
    main()
