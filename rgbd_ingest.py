"""RGB-D capture session (iPhone ARKit / Android ARCore, docs/rgbd_format.md) -> TSDF mesh.

    python rgbd_ingest.py captures/session_20261024-1310 scans/house            # folder
    python rgbd_ingest.py captures/session_20261024-1310.zip scans/house        # zip
    python rgbd_ingest.py captures/session_x scans/house --label --label-every 5  # + Gemma objects
    python rgbd_ingest.py --serve                      # live: ws://0.0.0.0:8770/ingest
    python rgbd_ingest.py path/to/roomscan/rgbd scans/venue   # RoomScan RGB-D export (frames.jsonl)
    python map_clean.py scans/house maps/house          # then: clean floor plan

Outputs in <out>: fused.ply (triangle mesh, Z-up map frame, metres), fused_points.ply
(TSDF point cloud), ingest.json (stats), objects.json (with --label).

iOS app specifics handled: depth PNGs are always read as uint16 millimetres (session.json
says depth_units "meters" regardless); `files` exists only in streamed per-frame JSON (it
is stripped before saving NNNNNN.json); zips wrap a session_.../ folder (maybe "_2"
suffixed); intrinsics_depth = RGB intrinsics scaled by size ratios; the live stream may
skip keyframe indices and does not reconnect.

Frames: ARKit/ARCore poses are camera-to-world, world Y-up, camera looks down -Z with
+Y up (OpenGL). open3d wants camera +Z forward, +Y down (OpenCV), so
    T_cv_zup = W @ T_gl @ diag(1, -1, -1, 1),  W: (x, y, z)_yup -> (x, -z, y)_zup
(the same Y-up -> Z-up rotation scan_map.py uses for iPhone exports).

Gemma 4 box convention (probed 2026-10-24 on gemma4:e4b-it-qat with synthetic images of
known boxes, both when told the convention and when not): box_2d = [ymin, xmin, ymax, xmax]
normalised 0-1000 per axis (independently of aspect ratio), typically within ~5% of truth.
normalise_box() still guards against pixel units and swapped corners.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOXEL = 0.05
SDF_TRUNC = 0.15
MAX_DEPTH = 4.0
MASK_DILATE_PX = 10
GEMMA_MODEL = "gemma4:e4b-it-qat"
CATEGORIES = ["structure", "movable_furniture", "clutter", "person"]

W_YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def pose_to_cv_zup(pose) -> np.ndarray:
    """ARKit/ARCore camera-to-world (GL camera, Y-up world) -> open3d camera-to-world (Z-up)."""
    return W_YUP_TO_ZUP @ np.asarray(pose, float).reshape(4, 4) @ GL_TO_CV


# ---------------- reading sessions ----------------

@dataclass
class Frame:
    index: int
    meta: dict
    rgb: np.ndarray                 # HxWx3 uint8 (RGB resolution)
    depth_mm: np.ndarray            # uint16, depth resolution
    conf: np.ndarray | None = None  # uint8 0/1/2
    mask: np.ndarray | None = None  # uint8, 255 = person
    rgb_path: str | None = None
    T: np.ndarray = field(default=None)  # camera-to-world, open3d camera, Z-up

    def __post_init__(self):
        if self.T is None:
            self.T = pose_to_cv_zup(self.meta["pose"])


def _imread(path, flags):
    import cv2

    if not os.path.exists(path):
        return None
    im = cv2.imread(path, flags)
    if im is None:
        raise ValueError(f"unreadable image {path}")
    return im


def load_frame(frames_dir: str, stem: str) -> Frame:
    import cv2

    meta = json.load(open(os.path.join(frames_dir, f"{stem}.json")))
    bgr = _imread(os.path.join(frames_dir, f"{stem}.jpg"), cv2.IMREAD_COLOR)
    depth = _imread(os.path.join(frames_dir, f"{stem}_depth.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise FileNotFoundError(f"frame {stem}: missing jpg or depth png")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    conf = _imread(os.path.join(frames_dir, f"{stem}_conf.png"), cv2.IMREAD_UNCHANGED)
    mask = _imread(os.path.join(frames_dir, f"{stem}_mask.png"), cv2.IMREAD_UNCHANGED)
    return Frame(int(meta.get("index", int(stem))), meta, bgr[..., ::-1].copy(), depth, conf, mask,
                 rgb_path=os.path.join(frames_dir, f"{stem}.jpg"))


def find_session_root(path: str) -> str:
    """Directory holding session.json (our format; a zip may wrap it in session_<ts>/), or
    frames.jsonl (RoomScan RGB-D export)."""
    for marker in ("session.json", "frames.jsonl"):
        if os.path.exists(os.path.join(path, marker)):
            return path
    for marker in ("session.json", "frames.jsonl"):
        hits = sorted(h for h in glob.glob(os.path.join(path, "**", marker), recursive=True) if "__MACOSX" not in h)
        if hits:
            return os.path.dirname(hits[0])
    sys.exit(f"no session.json (or RoomScan frames.jsonl) under {path}")


# ---------------- RoomScan RGB-D export (test path for real captures) ----------------
# rgbd/frames.jsonl, color/NNNNNN.jpg (960x720), depth/NNNNNN.u16 (raw LE uint16 mm, 256x192),
# conf/NNNNNN.u8 (raw 0/1/2). camera_to_world = 16 floats, simd column-major (ARKit
# camera.transform); intrinsics [fx, fy, cx, cy] for the colour image. Recorded at 10 Hz,
# no person masks: subsampled here to the keyframe policy (>= 0.15 m or >= 10 deg).

def _keyframe(prev, T, min_move=0.15, min_rot=10.0):
    if prev is None:
        return True
    if np.linalg.norm(T[:3, 3] - prev[:3, 3]) >= min_move:
        return True
    c = (np.trace(prev[:3, :3].T @ T[:3, :3]) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) >= min_rot


def iter_roomscan(root: str, keyframes=True):
    import cv2

    base = os.path.dirname(root.rstrip("/")) if not os.path.isdir(os.path.join(root, "color")) else root
    for d in (root, base):
        if os.path.isdir(os.path.join(d, "color")):
            base = d
            break
    prev = None
    for line in open(os.path.join(root, "frames.jsonl")):
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("tracking") != "normal":
            continue
        pose = np.array(j["camera_to_world"], float).reshape(4, 4).T     # column-major -> matrix
        if keyframes and not _keyframe(prev, pose):
            continue
        stem = f"{int(j['index']):06d}"
        (W, H), (w, h) = j.get("image_size", [960, 720]), j.get("depth_size", [256, 192])
        try:
            depth = np.fromfile(os.path.join(base, "depth", stem + ".u16"), dtype="<u2").reshape(h, w)
            cp = os.path.join(base, "conf", stem + ".u8")
            conf = np.fromfile(cp, dtype=np.uint8).reshape(h, w) if os.path.exists(cp) else None
            jpg = os.path.join(base, "color", stem + ".jpg")
            bgr = cv2.imread(jpg, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(jpg)
        except (OSError, ValueError) as e:
            print(f"warning: skipping RoomScan frame {stem}: {e}", file=sys.stderr)
            continue
        fx, fy, cx, cy = j["intrinsics"]
        meta = {"index": int(j["index"]), "timestamp": j.get("timestamp"), "pose": pose.tolist(),
                "intrinsics_rgb": dict(fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H),
                "intrinsics_depth": dict(fx=fx * w / W, fy=fy * h / H, cx=cx * w / W, cy=cy * h / H,
                                         width=w, height=h),
                "tracking": "normal"}
        prev = pose
        yield Frame(meta["index"], meta, bgr[..., ::-1].copy(), depth, conf, None, rgb_path=jpg)


def unzip_session(zpath: str, dest: str | None = None) -> str:
    dest = dest or tempfile.mkdtemp(prefix="rgbd_session_")
    with zipfile.ZipFile(zpath) as z:
        root = os.path.realpath(dest)
        for n in z.namelist():  # refuse path traversal
            if not os.path.realpath(os.path.join(dest, n)).startswith(root + os.sep):
                sys.exit(f"unsafe path in zip: {n}")
        z.extractall(dest)
    return find_session_root(dest)


def iter_session(root: str):
    frames_dir = os.path.join(root, "frames")
    stems = sorted({os.path.basename(p)[:-5] for p in glob.glob(os.path.join(frames_dir, "*.json"))},
                   key=lambda s: int(s) if s.isdigit() else s)
    for s in stems:
        try:
            yield load_frame(frames_dir, s)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"warning: skipping frame {s}: {e}", file=sys.stderr)


# ---------------- fusion ----------------

def clean_depth(f: Frame, max_depth=MAX_DEPTH, mask_dilate=MASK_DILATE_PX, use_mask=True) -> np.ndarray:
    """uint16 mm depth with persons (dilated mask), low confidence and far points zeroed."""
    import cv2

    d = f.depth_mm.copy()
    h, w = d.shape
    if f.conf is not None:
        c = f.conf if f.conf.shape == d.shape else cv2.resize(f.conf, (w, h), interpolation=cv2.INTER_NEAREST)
        d[c < 1] = 0
    if use_mask and f.mask is not None:
        m = f.mask if f.mask.ndim == 2 else f.mask[..., 0]
        if m.shape != d.shape:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m >= 128).astype(np.uint8)
        if mask_dilate > 0 and m.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_dilate + 1, 2 * mask_dilate + 1))
            m = cv2.dilate(m, k)
        d[m > 0] = 0
    d[d > max_depth * 1000] = 0
    return d


class Fuser:
    def __init__(self, voxel=VOXEL, sdf_trunc=SDF_TRUNC, max_depth=MAX_DEPTH, mask_dilate=MASK_DILATE_PX):
        import open3d as o3d

        self.o3d = o3d
        self.max_depth, self.mask_dilate = max_depth, mask_dilate
        self.vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.n_fused = self.n_skipped = self.masked_px = 0
        self.cam_xy = []

    def add(self, f: Frame) -> bool:
        import cv2

        o3d = self.o3d
        if f.meta.get("tracking", "normal") != "normal":
            self.n_skipped += 1
            return False
        K = f.meta["intrinsics_depth"]
        h, w = f.depth_mm.shape
        if (K.get("width", w), K.get("height", h)) != (w, h):  # trust the image, rescale K
            sx, sy = w / K["width"], h / K["height"]
            K = dict(fx=K["fx"] * sx, fy=K["fy"] * sy, cx=K["cx"] * sx, cy=K["cy"] * sy)
        d = clean_depth(f, self.max_depth, self.mask_dilate)
        self.masked_px += int(((f.depth_mm > 0) & (d == 0)).sum())
        rgb = cv2.resize(f.rgb, (w, h), interpolation=cv2.INTER_AREA)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(d),
            depth_scale=1000.0, depth_trunc=self.max_depth, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K["fx"], K["fy"], K["cx"], K["cy"])
        self.vol.integrate(rgbd, intr, np.linalg.inv(f.T))
        self.n_fused += 1
        self.cam_xy.append(f.T[:2, 3].tolist())
        return True

    def mesh(self):
        m = self.vol.extract_triangle_mesh()
        m.compute_vertex_normals()
        return m

    def points(self):
        return self.vol.extract_point_cloud()


# ---------------- Gemma labelling ----------------

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"label": {"type": "string"},
                       "category": {"type": "string", "enum": CATEGORIES},
                       "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
        "required": ["label", "category", "box_2d"]}}},
    "required": ["objects"]}

LABEL_PROMPT = (
    "You are mapping a home for a small walking robot. List the physical objects visible in this "
    "photo. For each give: label (short noun, e.g. 'wall', 'doorway', 'kitchen counter', 'couch', "
    "'chair', 'bag', 'shoes', 'box', 'person'); category, one of: structure (walls, doors, built-in "
    "counters/cabinets, stairs, columns), movable_furniture (couch, chair, table, bed, shelf), "
    "clutter (small items on the floor that are not normally there: bags, boxes, shoes, toys, "
    "laundry, bottles), person (any human, or body part); and box_2d = [ymin, xmin, ymax, xmax] "
    "normalised to 0-1000. Skip the floor and ceiling.")


def normalise_box(b, w, h):
    """-> (ymin, xmin, ymax, xmax) in 0-1 image fractions, or None."""
    try:
        y0, x0, y1, x1 = (float(v) for v in b[:4])
    except (TypeError, ValueError):
        return None
    if max(y0, x0, y1, x1) > 1000.5:          # pixel units
        y0, y1, x0, x1 = y0 / h * 1000, y1 / h * 1000, x0 / w * 1000, x1 / w * 1000
    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))
    y0, x0, y1, x1 = (min(max(v / 1000.0, 0.0), 1.0) for v in (y0, x0, y1, x1))
    if y1 - y0 < 0.01 or x1 - x0 < 0.01:
        return None
    return y0, x0, y1, x1


class Labeler:
    def __init__(self, model=GEMMA_MODEL, host=None, timeout=180):
        self.model, self.ok, self.calls, self.secs = model, True, 0, 0.0
        try:
            import ollama

            self.client = ollama.Client(host=host, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            print(f"warning: labelling disabled (ollama import failed: {e})", file=sys.stderr)
            self.ok = False

    def label(self, jpg_path: str) -> list[dict]:
        if not self.ok:
            return []
        t = time.time()
        try:
            r = self.client.chat(model=self.model, format=LABEL_SCHEMA, options={"temperature": 0},
                                 messages=[{"role": "user", "content": LABEL_PROMPT, "images": [jpg_path]}])
            out = json.loads(r["message"]["content"]).get("objects", [])
        except (ConnectionError, OSError) as e:
            print(f"warning: Ollama unreachable, labelling disabled: {e}", file=sys.stderr)
            self.ok = False
            return []
        except Exception as e:  # noqa: BLE001  (bad JSON, model error, ...): skip this frame
            print(f"warning: labelling {os.path.basename(jpg_path)} failed: {e}", file=sys.stderr)
            return []
        finally:
            self.calls += 1
            self.secs += time.time() - t
        return [o for o in out if isinstance(o, dict) and o.get("category") in CATEGORIES]


def lift_box(f: Frame, box01, band=0.6):
    """2D box (0-1 fractions) -> 3D axis-aligned box in the Z-up world, from the median depth
    of the box's central region and the pose. Returns (center, extent, n_points) or None."""
    K = f.meta["intrinsics_depth"]
    d = f.depth_mm.astype(np.float32) / 1000.0
    if f.conf is not None and f.conf.shape == d.shape:
        d[f.conf < 1] = 0
    d[d > MAX_DEPTH + 1] = 0
    h, w = d.shape
    y0, x0, y1, x1 = box01
    r0, r1 = int(y0 * h), max(int(y0 * h) + 1, int(math.ceil(y1 * h)))
    c0, c1 = int(x0 * w), max(int(x0 * w) + 1, int(math.ceil(x1 * w)))
    # median of the central 60% (less background leaking in at the box edges)
    rr0, rr1 = r0 + (r1 - r0) // 5, r1 - (r1 - r0) // 5
    cc0, cc1 = c0 + (c1 - c0) // 5, c1 - (c1 - c0) // 5
    core = d[rr0:max(rr1, rr0 + 1), cc0:max(cc1, cc0 + 1)]
    core = core[core > 0]
    if core.size < 5:
        return None
    med = float(np.median(core))
    vv, uu = np.mgrid[r0:r1, c0:c1]
    z = d[r0:r1, c0:c1]
    keep = (z > 0) & (np.abs(z - med) < max(band, 0.25 * med))
    if keep.sum() < 10:
        return None
    z, uu, vv = z[keep], uu[keep].astype(float), vv[keep].astype(float)
    pc = np.stack([(uu - K["cx"]) / K["fx"] * z, (vv - K["cy"]) / K["fy"] * z, z, np.ones_like(z)])
    pw = (f.T @ pc)[:3].T
    lo, hi = np.percentile(pw, 5, axis=0), np.percentile(pw, 95, axis=0)
    return (lo + hi) / 2, np.maximum(hi - lo, 0.05), int(keep.sum())


def _overlap(a, b):
    lo = np.maximum(a["min"], b["min"])
    hi = np.minimum(a["max"], b["max"])
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va, vb = float(np.prod(a["max"] - a["min"])), float(np.prod(b["max"] - b["min"]))
    return inter / max(1e-9, min(va, vb))


def _norm_label(s: str) -> str:
    s = re.sub(r"[^a-z ]", "", s.lower()).strip()
    return re.sub(r"(es|s)$", "", s) if len(s) > 3 else s


def merge_objects(dets: list[dict]) -> list[dict]:
    """Merge per-frame detections of the same label whose 3D boxes overlap (or nearly touch)."""
    groups: list[dict] = []
    for d in dets:
        mn, mx = np.array(d["center"]) - np.array(d["extent"]) / 2, np.array(d["center"]) + np.array(d["extent"]) / 2
        cand = {"min": mn, "max": mx}
        best = None
        for g in groups:
            if g["key"] != _norm_label(d["label"]):
                continue
            close = np.linalg.norm((g["min"] + g["max"]) / 2 - (mn + mx) / 2) < 0.35
            if _overlap(g, cand) > 0.25 or close:
                best = g
                break
        if best is None:
            groups.append({"key": _norm_label(d["label"]), "label": d["label"], "min": mn, "max": mx,
                           "cats": [d["category"]], "frames": [d["frame"]], "boxes": [(mn, mx)]})
        else:
            best["min"], best["max"] = np.minimum(best["min"], mn), np.maximum(best["max"], mx)
            best["cats"].append(d["category"])
            best["frames"].append(d["frame"])
            best["boxes"].append((mn, mx))
    out = []
    for g in groups:
        # robust box: median of the per-frame boxes when seen >= 3 times, union otherwise
        if len(g["boxes"]) >= 3:
            mn = np.median([b[0] for b in g["boxes"]], axis=0)
            mx = np.median([b[1] for b in g["boxes"]], axis=0)
        else:
            mn, mx = g["min"], g["max"]
        cat = max(set(g["cats"]), key=g["cats"].count)
        out.append({"label": g["label"], "category": cat, "center": ((mn + mx) / 2).round(3).tolist(),
                    "extent": (mx - mn).round(3).tolist(), "frames": sorted(set(g["frames"])),
                    "n_obs": len(g["frames"])})
    return out


def run_labelling(frames: list[Frame], every: int, model: str, out: str, host=None) -> list[dict]:
    lab = Labeler(model, host)
    dets, raw = [], []
    sel = [f for f in frames if f.meta.get("tracking", "normal") == "normal"][::max(1, every)]
    print(f"labelling {len(sel)} keyframes with {model} ...")
    for f in sel:
        if not lab.ok:
            break
        Kr = f.meta.get("intrinsics_rgb", {})
        w, h = Kr.get("width", f.rgb.shape[1]), Kr.get("height", f.rgb.shape[0])
        objs = lab.label(f.rgb_path)
        for o in objs:
            box = normalise_box(o.get("box_2d"), w, h)
            raw.append({"frame": f.index, **o})
            if box is None or o["category"] == "structure":
                continue  # structure stays in the map whatever it is; no need to localise it
            lifted = lift_box(f, box)
            if lifted is None:
                continue
            c, e, n = lifted
            if o["category"] in ("clutter", "person") and (e.max() > 2.5 or e[:2].max() > 1.6):
                continue  # implausibly big: depth leaked from the background
            dets.append({"label": o["label"], "category": o["category"], "center": c.tolist(),
                         "extent": e.tolist(), "frame": f.index})
    objects = merge_objects(dets)
    json.dump({"model": model, "box_convention": "[ymin, xmin, ymax, xmax] 0-1000",
               "frame": "Z-up map frame of fused.ply (before map_clean levelling)",
               "labelled_frames": [f.index for f in sel], "calls": lab.calls,
               "seconds": round(lab.secs, 1), "ok": lab.ok,
               "objects": objects, "raw_detections": raw},
              open(os.path.join(out, "objects.json"), "w"), indent=2)
    cats = {c: sum(o["category"] == c for o in objects) for c in CATEGORIES}
    print(f"objects.json: {len(objects)} objects {cats} from {len(raw)} detections "
          f"({lab.calls} calls, {lab.secs:.0f}s)" + ("" if lab.ok else "  [Ollama failed: partial/empty]"))
    return objects


# ---------------- batch ingest ----------------

def write_outputs(fuser: Fuser, out: str, extra: dict | None = None):
    import open3d as o3d

    os.makedirs(out, exist_ok=True)
    mesh = fuser.mesh()
    pcd = fuser.points()
    if len(pcd.points):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_triangle_mesh(os.path.join(out, "fused.ply"), mesh)
    o3d.io.write_point_cloud(os.path.join(out, "fused_points.ply"), pcd)
    info = {"frames_fused": fuser.n_fused, "frames_skipped_tracking": fuser.n_skipped,
            "person_or_lowconf_px_dropped": fuser.masked_px, "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles), "points": len(pcd.points),
            "camera_xy": fuser.cam_xy, "voxel": VOXEL, "sdf_trunc": SDF_TRUNC, "max_depth": MAX_DEPTH,
            **(extra or {})}
    json.dump(info, open(os.path.join(out, "ingest.json"), "w"), indent=1)
    print(f"fused {fuser.n_fused} frames (skipped {fuser.n_skipped} non-normal tracking): "
          f"{len(mesh.vertices)} verts / {len(mesh.triangles)} tris -> {out}/fused.ply")
    return mesh


def ingest(src: str, out: str, label=False, label_every=5, model=GEMMA_MODEL, host=None):
    tmp = None
    if src.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="rgbd_zip_")
        root = unzip_session(src, tmp)
    else:
        root = find_session_root(src)
    if os.path.exists(os.path.join(root, "session.json")):
        session = json.load(open(os.path.join(root, "session.json")))
        frames_iter = iter_session(root)
    else:
        session = {"device": "iphone", "model": "RoomScan RGB-D export", "coord": "arkit_y_up",
                   "has_person_mask": False}
        frames_iter = iter_roomscan(root)
    print(f"session {root}: {session.get('device')} {session.get('model')} coord={session.get('coord')}")
    fuser = Fuser()
    frames = []
    for f in frames_iter:
        fuser.add(f)
        if label:
            f.rgb = f.rgb[::8, ::8].copy()  # keep only what labelling needs: path + depth + pose
            frames.append(f)
    os.makedirs(out, exist_ok=True)
    write_outputs(fuser, out, {"session": session, "source": os.path.abspath(src)})
    if label:
        try:
            run_labelling(frames, label_every, model, out, host)
        except Exception as e:  # noqa: BLE001  labelling must never break ingest
            print(f"warning: labelling failed, continuing without objects.json: {e}", file=sys.stderr)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- live WebSocket ingest ----------------

SUFFIX = {"rgb": ".jpg", "depth": "_depth.png", "conf": "_conf.png", "mask": "_mask.png"}


def _kind(name: str, pos: int) -> str:
    n = name.lower()
    for k in ("depth", "conf", "mask"):
        if k in n:
            return k
    if n.endswith((".jpg", ".jpeg")) or n in ("rgb", "color", "image"):
        return "rgb"
    return ["rgb", "depth", "conf", "mask"][min(pos, 3)]


def preview_png(fuser: Fuser, path: str):
    """Coarse top-down occupancy of the TSDF so far (floor/obstacle/unknown)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from map_clean import grid_from_points

    pts = np.asarray(fuser.points().points)
    if len(pts) < 500:
        return
    g = grid_from_points(pts, res=0.10)
    H, W = g["free"].shape
    img = np.full((H, W, 3), 0.55)
    img[g["free"]] = (0.96, 0.96, 0.94)
    img[g["obstacle"]] = (0.2, 0.2, 0.25)
    ox, oy, r = g["origin"][0], g["origin"][1], g["res"]
    fig, ax = plt.subplots(figsize=(6, 6 * H / max(W, 1) + 0.5))
    ax.imshow(img, origin="lower", extent=[ox, ox + W * r, oy, oy + H * r], interpolation="nearest")
    if fuser.cam_xy:
        ax.plot(*np.array(fuser.cam_xy).T, ".-", color="#e08a1e", ms=3, lw=1)
    ax.set_title(f"live preview: {fuser.n_fused} keyframes")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _rerun():
    try:
        import rerun as rr  # optional

        rr.init("rgbd_ingest", spawn=True)
        return rr
    except Exception:  # noqa: BLE001
        return None


async def handle_ingest(ws, captures: str, scans: str, preview_every: int, label: bool, label_every: int):
    path = getattr(getattr(ws, "request", None), "path", "/ingest")
    if not path.startswith("/ingest"):
        await ws.close(code=1008, reason="use /ingest")
        return
    session = json.loads(await ws.recv())
    name = "session_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.join(captures, name)
    os.makedirs(os.path.join(root, "frames"), exist_ok=True)
    json.dump(session, open(os.path.join(root, "session.json"), "w"), indent=2)
    out = os.path.join(scans, name)
    os.makedirs(out, exist_ok=True)
    print(f"[live] {ws.remote_address}: {session.get('device')} -> {root}")
    fuser, rr = Fuser(), _rerun()
    frames, since = [], 0
    while True:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            print("[live] warning: unexpected binary message, ignored", file=sys.stderr)
            continue
        meta = json.loads(msg)
        if meta.get("end"):
            break
        idx = int(meta["index"])
        stem = f"{idx:06d}"
        names = meta.pop("files", ["rgb", "depth", "conf", "mask"])
        for pos, n in enumerate(names):
            data = await ws.recv()
            if isinstance(data, str):
                raise ValueError(f"frame {idx}: expected binary for {n}, got text")
            with open(os.path.join(root, "frames", stem + SUFFIX[_kind(n, pos)]), "wb") as fh:
                fh.write(data)
        json.dump(meta, open(os.path.join(root, "frames", stem + ".json"), "w"))
        try:
            f = load_frame(os.path.join(root, "frames"), stem)
        except Exception as e:  # noqa: BLE001
            print(f"[live] warning: frame {idx} unreadable: {e}", file=sys.stderr)
            continue
        if await asyncio.to_thread(fuser.add, f):
            since += 1
            if rr is not None:
                rr.log("world/camera", rr.Points3D([f.T[:3, 3]]))
        if label:
            f.rgb = f.rgb[::8, ::8].copy()
            frames.append(f)
        if since >= preview_every:
            since = 0
            await asyncio.to_thread(preview_png, fuser, os.path.join(out, "preview.png"))
            print(f"[live] {fuser.n_fused} keyframes fused -> {out}/preview.png")
            if rr is not None:
                p = fuser.points()
                rr.log("world/tsdf", rr.Points3D(np.asarray(p.points), colors=np.asarray(p.colors)))
    print(f"[live] session end: {fuser.n_fused} fused")
    await asyncio.to_thread(write_outputs, fuser, out, {"session": session, "source": root})
    await asyncio.to_thread(preview_png, fuser, os.path.join(out, "preview.png"))
    if label:
        try:
            await asyncio.to_thread(run_labelling, frames, label_every, GEMMA_MODEL, out)
        except Exception as e:  # noqa: BLE001
            print(f"[live] warning: labelling failed: {e}", file=sys.stderr)
    try:
        await ws.send(json.dumps({"ok": True, "session": root, "out": out, "fused": fuser.n_fused}))
    except Exception:  # noqa: BLE001
        pass


async def serve(host="0.0.0.0", port=8770, captures="captures", scans="scans", preview_every=10,
                label=False, label_every=5, stop: asyncio.Event | None = None):
    from websockets.asyncio.server import serve as ws_serve

    async def handler(ws):
        try:
            await handle_ingest(ws, captures, scans, preview_every, label, label_every)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[live] session aborted: {e}", file=sys.stderr)

    async with ws_serve(handler, host, port, max_size=64 * 2 ** 20, ping_timeout=120):
        print(f"[live] listening on ws://{host}:{port}/ingest  (captures -> {captures}/, output -> {scans}/)")
        await (stop.wait() if stop else asyncio.Future())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="session folder or .zip")
    ap.add_argument("out", nargs="?", help="output dir (default scans/<session name>)")
    ap.add_argument("--serve", action="store_true", help="live WebSocket ingest on :8770/ingest")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--captures", default=os.path.join(HERE, "captures"))
    ap.add_argument("--scans", default=os.path.join(HERE, "scans"))
    ap.add_argument("--preview-every", type=int, default=10)
    ap.add_argument("--label", action="store_true", help="label keyframes with Gemma 4 via Ollama")
    ap.add_argument("--label-every", type=int, default=5)
    ap.add_argument("--model", default=GEMMA_MODEL)
    ap.add_argument("--ollama-host", default=None)
    a = ap.parse_args()
    if a.serve:
        asyncio.run(serve(a.host, a.port, a.captures, a.scans, a.preview_every, a.label, a.label_every))
        return
    if not a.session:
        ap.error("give a session folder/.zip, or --serve")
    name = os.path.basename(a.session.rstrip("/"))
    name = name[:-4] if name.lower().endswith(".zip") else name
    ingest(a.session, a.out or os.path.join(a.scans, name), a.label, a.label_every, a.model, a.ollama_host)


if __name__ == "__main__":
    main()
