#!/usr/bin/env python3
"""Demo video renderer: iPhone RGB-D walkthrough -> bird's-eye map + path + RGB/depth.

    .venv/bin/python make_video.py --session captures/session_20261024-1347.zip \
        --map maps/agihouse --scan scans/agihouse/fused.ply --out videos

Outputs (in --out):
    walkthrough_1080p.mp4   1920x1080 30 fps, title card, live walkthrough, summary card
    twin_orbit_1080p.mp4    orbit around the fused point cloud with the path in 3D (--orbit)
    highlight_10s.mp4/.gif  10 s cut for the README
    stills/*.png            best moments for slides

Frames (same as rgbd_ingest.py + map_clean.py):
    ARKit pose (camera-to-world, Y-up, GL camera)  --W_YUP_TO_ZUP-->  ingest Z-up frame
    --level_transform (maps/<name>/clean_report.json, map_clean.py step 1)-->  map frame
Frames are rendered with numpy/OpenCV/PIL in worker processes and piped to ffmpeg as raw RGB.
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
W_OUT, H_OUT, FPS = 1920, 1080, 30

# same constants as rgbd_ingest.py
W_YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# ---------------- palette ----------------
BG = np.array([9, 11, 16], np.float32)
PANEL = (19, 22, 30)
TEXT = (242, 244, 248)
TEXT2 = (140, 148, 168)
TEXT3 = (92, 100, 120)
CYAN = np.array([34, 211, 238], np.float32)
VIOLET = np.array([139, 92, 246], np.float32)
PINK = np.array([244, 63, 160], np.float32)
FLOOR = np.array([231, 227, 220], np.float32)
WALL = np.array([47, 58, 76], np.float32)
MAGENTA = np.array([255, 45, 170], np.float32)
AMBER = np.array([251, 191, 36], np.float32)
GREEN = (52, 211, 153)

FONT_PATH = "/System/Library/Fonts/SFNS.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO_PATH = "/System/Library/Fonts/SFNSMono.ttf"


@functools.lru_cache(maxsize=None)
def font(size: int, weight: int = 400, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = MONO_PATH if mono and os.path.exists(MONO_PATH) else FONT_PATH
    f = ImageFont.truetype(path, size)
    try:
        axes = f.get_variation_axes()
        vals = []
        for a in axes:
            n = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
            if n == "Weight":
                vals.append(weight)
            elif n == "Optical Size":
                vals.append(max(a["minimum"], min(a["maximum"], size)))
            else:
                vals.append(a["default"])
        f.set_variation_by_axes(vals)
    except Exception:
        pass
    return f


def grad_color(u: float) -> np.ndarray:
    """cyan -> violet -> pink for u in [0, 1]."""
    u = float(np.clip(u, 0, 1))
    if u < 0.5:
        return CYAN + (VIOLET - CYAN) * (u / 0.5)
    return VIOLET + (PINK - VIOLET) * ((u - 0.5) / 0.5)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def ease_out_back(t, s=1.9):
    t = np.clip(t, 0, 1) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def fmt_time(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


# ---------------- session ----------------

def open_session(path: str) -> str:
    """Return the session folder (the one holding frames/); unzip zips into a temp dir."""
    if path.endswith(".zip"):
        dest = tempfile.mkdtemp(prefix="mkvid_")
        import atexit; atexit.register(shutil.rmtree, dest, True)
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        path = dest
    for root, dirs, _ in os.walk(path):
        if "frames" in dirs:
            return root
    sys.exit(f"no frames/ folder under {path}")


def load_level(map_dir: str) -> np.ndarray:
    rep = os.path.join(map_dir, "clean_report.json")
    if os.path.exists(rep):
        js = json.load(open(rep))
        if "level_transform" in js:
            return np.array(js["level_transform"], float)
    # not saved: recompute exactly like map_clean.clean step 1
    sys.path.insert(0, HERE)
    import open3d as o3d
    import map_clean
    ingest_dir = os.path.join(HERE, "scans", os.path.basename(os.path.normpath(map_dir)))
    mesh = o3d.io.read_triangle_mesh(os.path.join(ingest_dir, "fused.ply"))
    n, d, _ = map_clean.find_floor(np.asarray(mesh.vertices))
    T = np.eye(4)
    T[:3, :3] = map_clean._rot_a_to_b(n, np.array([0, 0, 1.0]))
    T[2, 3] = d
    return T


def load_frames(root: str, L: np.ndarray):
    fr = os.path.join(root, "frames")
    stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(fr, "*.json")))
    metas = [json.load(open(os.path.join(fr, s + ".json"))) for s in stems]
    ts = np.array([m["timestamp"] for m in metas], float)
    # strictly increasing for interpolation
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1e-3
    T_gl = np.array([L @ W_YUP_TO_ZUP @ np.array(m["pose"], float).reshape(4, 4) for m in metas])
    return fr, stems, metas, ts, T_gl


def image_rotation(T_gl: np.ndarray) -> int:
    """np.rot90 k that makes the sensor-landscape image upright (world up -> image top)."""
    up = np.median(np.einsum("nij,j->ni", np.transpose(T_gl[:, :3, :3], (0, 2, 1)), [0, 0, 1.0]), axis=0)
    x, y = up[0], up[1]  # in GL camera coords: +x image right, +y image up
    if abs(x) > abs(y):
        return -1 if x < 0 else 1  # up is image-left -> rotate clockwise
    return 0 if y > 0 else 2


# ---------------- map ----------------

def dominant_angle(free: np.ndarray, obstacle: np.ndarray) -> float:
    """Dominant wall direction (radians, mod 90 deg) from edge gradients of the grid."""
    known = (free | obstacle).astype(np.float32)
    img = cv2.GaussianBlur(np.maximum(known * 0.5, obstacle.astype(np.float32)), (0, 0), 1.5)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi / 2)
    sel = mag > np.percentile(mag[mag > 0], 50)
    hist, edges = np.histogram(ang[sel], bins=180, range=(0, np.pi / 2), weights=mag[sel])
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    hist = np.convolve(np.concatenate([hist[-6:], hist, hist[:6]]), k, "same")[6:-6]
    return float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)


class MapView:
    """World (map frame, metres) <-> display pixels for a rotated, axis-aligned floor plan."""

    def __init__(self, phi, ctr, scale, px0, py0):
        self.phi, self.ctr, self.s, self.px0, self.py0 = phi, np.asarray(ctr, float), scale, px0, py0
        c, s = math.cos(-phi), math.sin(-phi)
        self.R = np.array([[c, -s], [s, c]])

    def rot(self, xy):
        return (np.asarray(xy, float) - self.ctr) @ self.R.T

    def __call__(self, xy):
        uv = self.rot(xy)
        out = np.empty_like(uv)
        out[..., 0] = self.px0 + self.s * uv[..., 0]
        out[..., 1] = self.py0 - self.s * uv[..., 1]
        return out

    def cell_affine(self, origin, res, ss=1.0):
        """2x3 affine: grid (col, row) -> display px (scaled by ss)."""
        A = np.diag([1.0, -1.0]) @ (self.s * self.R) * res
        b = np.array([self.px0, self.py0]) + np.diag([1.0, -1.0]) @ (self.s * self.R) @ (
            np.asarray(origin) + 0.5 * res - self.ctr)
        return np.hstack([A, b[:, None]]) * ss


def fit_view(free, obstacle, origin, res, phi0, box_w, box_h, box_x, box_y, prefer_landscape=True):
    known = free | obstacle
    rr, cc = np.nonzero(known)
    pts = np.stack([origin[0] + (cc + 0.5) * res, origin[1] + (rr + 0.5) * res], 1)
    best = None
    for k in range(4):
        phi = phi0 + k * np.pi / 2
        c, s = math.cos(-phi), math.sin(-phi)
        uv = pts @ np.array([[c, -s], [s, c]]).T
        lo, hi = np.percentile(uv, 0.2, axis=0), np.percentile(uv, 99.8, axis=0)
        span = hi - lo
        sc = min(box_w / span[0], box_h / span[1])
        key = (round(sc, 3), -k)
        if best is None or key > best[0]:
            ctr_uv = (lo + hi) / 2
            ctr = np.array([[c, s], [-s, c]]) @ ctr_uv  # inverse rotation
            best = (key, phi, ctr, sc)
    _, phi, ctr, sc = best
    return MapView(phi, ctr, sc, box_x + box_w / 2, box_y + box_h / 2)


def clean_mask(m: np.ndarray, min_cells: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_cells
    return keep[lab]


def soft_warp(mask: np.ndarray, M: np.ndarray, size, blur=0.8) -> np.ndarray:
    """Warp a boolean cell mask to display with smooth anti-aliased edges (float 0..1)."""
    f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), blur)
    w = cv2.warpAffine(f, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
    return smoothstep(0.38, 0.62, w).astype(np.float32)


# ---------------- coverage (first-seen time per cell) ----------------

def coverage_times(fr, stems, metas, ts, T_gl, origin, res, shape, known):
    H, W = shape
    first = np.full(shape, np.inf, np.float32)
    person = np.zeros(len(stems), np.float32)
    step = 4
    for i, s in enumerate(stems):
        d = cv2.imread(os.path.join(fr, s + "_depth.png"), cv2.IMREAD_UNCHANGED)
        mk = cv2.imread(os.path.join(fr, s + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        cf = cv2.imread(os.path.join(fr, s + "_conf.png"), cv2.IMREAD_GRAYSCALE)
        if mk is not None:
            person[i] = float((mk > 127).mean())
        if d is None:
            continue
        intr = metas[i]["intrinsics_depth"]
        z = d[::step, ::step].astype(np.float32) / 1000.0
        v, u = np.mgrid[0:d.shape[0]:step, 0:d.shape[1]:step]
        ok = (z > 0.1) & (z < 4.0)
        if mk is not None:
            ok &= mk[::step, ::step] < 128
        if cf is not None:
            ok &= cf[::step, ::step] >= 1
        z, u, v = z[ok], u[ok], v[ok]
        X = (u - intr["cx"]) / intr["fx"] * z
        Y = (v - intr["cy"]) / intr["fy"] * z
        P = np.stack([X, Y, z, np.ones_like(z)], 0)
        Tw = T_gl[i] @ GL_TO_CV
        pw = (Tw @ P)[:3].T
        pw = pw[pw[:, 2] < 2.2]
        cam = T_gl[i][:3, 3]
        # points and the free space along their rays
        fr_s = np.array([0.35, 0.6, 0.85, 1.0])
        q = cam[None, None, :2] + (pw[None, :, :2] - cam[None, None, :2]) * fr_s[:, None, None]
        q = q.reshape(-1, 2)
        c = ((q[:, 0] - origin[0]) / res).astype(int)
        r = ((q[:, 1] - origin[1]) / res).astype(int)
        inb = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        np.minimum.at(first, (r[inb], c[inb]), ts[i])
    # cells never hit: time of first pass within 2.5 m, else the end
    miss = known & ~np.isfinite(first)
    if miss.any():
        from scipy.spatial import cKDTree
        rr, cc = np.nonzero(miss)
        cxy = np.stack([origin[0] + (cc + 0.5) * res, origin[1] + (rr + 0.5) * res], 1)
        tree = cKDTree(T_gl[:, :2, 3])
        hits = tree.query_ball_point(cxy, 2.5)
        first[rr, cc] = [ts[min(h)] if h else ts[-1] for h in hits]
    first[~np.isfinite(first)] = ts[-1]
    first = cv2.medianBlur(first, 3)
    return first, person


# ---------------- landmarks ----------------

def load_landmarks(map_dir, ts, T_gl, hfov):
    p = os.path.join(map_dir, "places.json")
    try:
        js = json.load(open(p))
    except Exception:
        return []
    items = []
    if isinstance(js, dict):
        if isinstance(js.get("places"), (list, dict)):
            js = js["places"]
    if isinstance(js, dict):
        for k, v in js.items():
            if isinstance(v, dict):
                items.append(dict(v, name=v.get("name", k)))
    elif isinstance(js, list):
        items = [v for v in js if isinstance(v, dict)]
    out = []
    cam = T_gl[:, :2, 3]
    fwd = -T_gl[:, :2, 2]
    for v in items:
        if v.get("kind") != "landmark" or "xy" not in v:
            continue
        xy = np.array(v["xy"][:2], float)
        f = v.get("first_seen_frame")
        if f is not None and 0 <= int(f) < len(ts):
            t = ts[int(f)]
        else:
            d = xy - cam
            dist = np.linalg.norm(d, axis=1)
            ang = np.abs(np.arctan2(fwd[:, 0] * d[:, 1] - fwd[:, 1] * d[:, 0], (fwd * d).sum(1)))
            vis = np.nonzero((dist < 3.5) & (ang < hfov / 2 + 0.15))[0]
            t = ts[vis[0]] if len(vis) else ts[int(np.argmin(dist))]
        name = str(v.get("landmark") or v.get("label") or v.get("name")).replace("_", " ").strip()
        name = name[:1].upper() + name[1:]
        name = " ".join(w.upper() if w.lower() in ("tv", "ai", "agi") else w for w in name.split())
        out.append({"name": name, "xy": xy, "t": float(t)})
    out.sort(key=lambda d: d["t"])
    kept = []
    for d in out:  # same name within 2 m of an earlier one -> duplicate sighting
        if any(k["name"] == d["name"] and np.linalg.norm(k["xy"] - d["xy"]) < 2.0 for k in kept):
            continue
        kept.append(d)
    return kept


def place_labels(S):
    """Static, collision-free label positions (earliest landmark wins); None = pin only."""
    x0, y0, x1, y1 = S.map_rect
    bx0, by0, bx1, by1 = S.box[0] - x0 - 20, S.box[1] - y0 - 30, S.box[2] - x0 + 20, S.box[3] - y0 + 20
    placed = []
    pins = [S.view(lm["xy"]) for lm in S.landmarks]
    for lm, p in zip(S.landmarks, pins):
        spr = chip(lm["name"], 14, 600, fg=TEXT, bg=(24, 28, 38, 235), padx=10, pady=6)
        lm["spr"] = spr
        h, w = spr.shape[:2]
        lm["lab"] = None
        cands = [(-w / 2, -14 - h), (-w / 2, 14), (14, -h / 2), (-14 - w, -h / 2),
                 (8, -10 - h), (-8 - w, -10 - h), (8, 10), (-8 - w, 10)]
        for dx, dy in cands:
            r = (p[0] + dx - 3, p[1] + dy - 3, p[0] + dx + w + 3, p[1] + dy + h + 3)
            if r[0] < bx0 or r[1] < by0 or r[2] > bx1 or r[3] > by1:
                continue
            if any(not (r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3]) for q in placed):
                continue
            if any(q is not lm and r[0] - 6 < pp[0] < r[2] + 6 and r[1] - 6 < pp[1] < r[3] + 6
                   for q, pp in zip(S.landmarks, pins) if q is not lm):
                continue
            placed.append(r)
            lm["lab"] = (dx, dy)
            break


# ---------------- drawing helpers ----------------

def rounded_mask(w, h, r, ss=4):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def blend(dst, color, alpha, x=0, y=0):
    """dst float32 HxWx3; alpha HxW (0..1) placed at (x, y), color (3,) or HxWx3."""
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None]
    c = color if np.ndim(color) == 1 else color[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = dst[y0:y1, x0:x1]
    reg += (np.asarray(c, np.float32) - reg) * a


def put_text(draw, xy, txt, size, weight=400, fill=TEXT, anchor="la", mono=False):
    draw.text(xy, txt, font=font(size, weight, mono), fill=tuple(int(c) for c in fill), anchor=anchor)


# ---------------- scene ----------------

class Scene:
    pass


def build_scene(args) -> Scene:
    S = Scene()
    t0 = time.time()
    S.root = open_session(args.session)
    S.L = load_level(args.map)
    S.fr, S.stems, S.metas, S.ts, S.T_gl = load_frames(S.root, S.L)
    n = len(S.stems)
    S.n = n
    S.rot_k = image_rotation(S.T_gl)
    intr = S.metas[0]["intrinsics_rgb"]
    # horizontal FOV of the upright image
    S.hfov = 2 * math.atan((intr["height"] if S.rot_k in (1, -1) else intr["width"]) / 2 / intr["fy"])
    S.session_dur = float(S.ts[-1])
    S.speed = args.speed or max(1, round(S.session_dur / args.target))
    S.main_dur = S.session_dur / S.speed
    S.t_title, S.t_end = args.title_secs, args.end_secs
    S.total = S.t_title + S.main_dur + S.t_end
    S.nframes = int(round(S.total * FPS))

    # ---- path (smoothed, interpolated) ----
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import PchipInterpolator
    xy = S.T_gl[:, :2, 3]
    xy_s = gaussian_filter1d(xy, 1.2, axis=0, mode="nearest")
    fwd = -S.T_gl[:, :2, 2]
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    yaw_s = gaussian_filter1d(yaw, 1.0, mode="nearest")
    S.pos_f = PchipInterpolator(S.ts, xy_s, axis=0)
    S.yaw_f = PchipInterpolator(S.ts, yaw_s)
    S.dense_t = np.linspace(0, S.session_dur, int(S.session_dur * 8) + 1)
    S.dense = S.pos_f(S.dense_t)
    seg = np.linalg.norm(np.diff(S.dense, axis=0), axis=1)
    S.dense_dist = np.concatenate([[0], np.cumsum(seg)])
    S.path_len = float(S.dense_dist[-1])

    # ---- grid ----
    g = np.load(os.path.join(args.map, "grid.npz"))
    meta = json.load(open(os.path.join(args.map, "meta.json")))
    S.free, S.obst = g["free"], g["obstacle"]
    S.origin, S.res = np.array(meta["origin"], float), float(meta["res"])
    c = ((xy[:, 0] - S.origin[0]) / S.res).astype(int)
    r = ((xy[:, 1] - S.origin[1]) / S.res).astype(int)
    inb = (r >= 0) & (r < S.free.shape[0]) & (c >= 0) & (c < S.free.shape[1])
    on = np.zeros(n, bool)
    on[inb] = S.free[r[inb], c[inb]]
    S.on_free_pct = 100.0 * on.mean()
    S.on_obst_pct = 100.0 * S.obst[r[inb], c[inb]].sum() / n
    S.area_free = float(S.free.sum() * S.res ** 2)
    print(f"session {n} keyframes, {S.session_dur:.1f} s, speed x{S.speed} -> {S.total:.1f} s video; "
          f"path {S.path_len:.1f} m; on free cells {S.on_free_pct:.1f}% (on obstacles {S.on_obst_pct:.1f}%)")

    # ---- coverage ----
    known = S.free | S.obst
    S.first, S.person = coverage_times(S.fr, S.stems, S.metas, S.ts, S.T_gl, S.origin, S.res,
                                       S.free.shape, known)
    S.person_on = S.person > 0.004
    S.landmarks = load_landmarks(args.map, S.ts, S.T_gl, S.hfov)
    print(f"coverage done, {len(S.landmarks)} landmarks ({time.time() - t0:.1f} s)")

    layout(S)
    place_labels(S)
    build_static(S)
    print(f"scene ready ({time.time() - t0:.1f} s)")
    return S


def layout(S):
    M = 56
    S.hdr_y = 40
    S.top, S.bot = 136, 972
    S.map_rect = (M, S.top, 1012, S.bot)                      # x0, y0, x1, y1
    rx0, rx1 = 1040, W_OUT - M
    pw = (rx1 - rx0 - 20) // 2
    ph = int(round(pw * 4 / 3)) if S.rot_k in (1, -1) else int(round(pw * 3 / 4))
    S.rgb_rect = (rx0, S.top, rx0 + pw, S.top + ph)
    S.dep_rect = (rx1 - pw, S.top, rx1, S.top + ph)
    S.stat_rect = (rx0, S.top + ph + 20, rx1, S.bot)
    S.foot_y = 1012
    # map fit box inside the map panel
    x0, y0, x1, y1 = S.map_rect
    S.box = (x0 + 30, y0 + 70, x1 - 30, y1 - 64)
    phi0 = dominant_angle(S.free, S.obst)
    S.view = fit_view(S.free, S.obst, S.origin, S.res, phi0, S.box[2] - S.box[0], S.box[3] - S.box[1],
                      S.box[0] - x0, S.box[1] - y0)
    S.phi_deg = math.degrees(S.view.phi)
    print(f"map rotation {S.phi_deg:.1f} deg, scale {S.view.s:.1f} px/m")


def build_static(S):
    """Static background frame and the pre-warped map layers (map-panel local coords)."""
    ss = 2
    bg = np.zeros((H_OUT, W_OUT, 3), np.float32) + BG
    yy, xx = np.mgrid[0:H_OUT, 0:W_OUT].astype(np.float32)
    glow = np.exp(-(((xx - 520) / 900) ** 2 + ((yy + 200) / 700) ** 2))
    bg += glow[..., None] * np.array([14, 18, 34], np.float32)
    glow2 = np.exp(-(((xx - 1800) / 700) ** 2 + ((yy - 1200) / 600) ** 2))
    bg += glow2[..., None] * np.array([22, 10, 30], np.float32)

    def panel(rect, r=22):
        x0, y0, x1, y1 = rect
        w, h = x1 - x0, y1 - y0
        m = rounded_mask(w, h, r)
        sh = np.zeros((h + 80, w + 80), np.float32)
        sh[40:40 + h, 40:40 + w] = m
        sh = cv2.GaussianBlur(sh, (0, 0), 16)
        blend(bg, np.zeros(3, np.float32), sh * 0.55, x0 - 40, y0 - 40 + 10)
        blend(bg, np.array(PANEL, np.float32), m, x0, y0)
        # 1 px top highlight border
        edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
        blend(bg, np.array([255, 255, 255], np.float32), np.clip(edge, 0, 1) * 0.07, x0, y0)
        return m

    S.map_mask = panel(S.map_rect)
    S.rgb_mask = rounded_mask(S.rgb_rect[2] - S.rgb_rect[0], S.rgb_rect[3] - S.rgb_rect[1], 18)
    S.dep_mask = S.rgb_mask
    panel(S.rgb_rect, 18)
    panel(S.dep_rect, 18)
    panel(S.stat_rect)

    # ---- map layers ----
    x0, y0, x1, y1 = S.map_rect
    mw, mh = x1 - x0, y1 - y0
    S.mw, S.mh = mw, mh
    M2 = S.view.cell_affine(S.origin, S.res, ss)
    known = clean_mask(S.free | S.obst, 40)
    known = cv2.morphologyEx(known.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    obst = clean_mask(S.obst & known, 6)
    size2 = (mw * ss, mh * ss)
    fa = soft_warp(known, M2, size2, 1.0)
    oa = soft_warp(obst, M2, size2, 0.7)
    down = lambda a: cv2.resize(a, (mw, mh), interpolation=cv2.INTER_AREA)
    fa, oa = down(fa), down(oa)
    # 1 m grid, axis aligned in the rotated frame
    gridl = np.zeros((mh, mw), np.float32)
    uv0 = S.view.rot(S.view.ctr)
    step = S.view.s
    ox = S.view.px0 % step
    oy = S.view.py0 % step
    for gx in np.arange(ox, mw, step):
        gridl[:, int(gx)] = 1
    for gy in np.arange(oy, mh, step):
        gridl[int(gy), :] = 1
    # floor shadow for depth
    sh = cv2.GaussianBlur(fa, (0, 0), 10)
    sh = np.roll(sh, 8, axis=0)
    S.lay_shadow = sh * 0.6
    # full-colour map (floor + grid + walls) and its alpha
    col = np.zeros((mh, mw, 3), np.float32) + FLOOR
    # gentle vignette on the floor
    yy, xx = np.mgrid[0:mh, 0:mw].astype(np.float32)
    vig = 1 - 0.06 * (((xx - mw / 2) / mw) ** 2 + ((yy - mh / 2) / mh) ** 2) * 4
    col *= vig[..., None]
    col += (np.array([200, 196, 190], np.float32) - col) * (gridl * 0.35)[..., None]
    # inner edge line on floor boundary
    edge = np.clip(fa - cv2.erode(fa, np.ones((3, 3), np.uint8)), 0, 1)
    col += (WALL - col) * (edge * 0.5)[..., None]
    col += (WALL - col) * oa[..., None]
    S.lay_col = col
    S.lay_a = np.clip(np.maximum(fa, oa), 0, 1)
    # ghost (not yet observed): faint blueprint
    ghost = np.clip(fa * 0.035 + edge * 0.09 + oa * 0.05, 0, 1)
    S.lay_ghost = ghost
    # reveal time (session seconds) per display pixel
    Mt = S.view.cell_affine(S.origin, S.res, 1.0)
    tm = S.first.copy()
    tm[~known] = S.ts[-1]
    tmap = cv2.warpAffine(tm, Mt, (mw, mh), flags=cv2.INTER_LINEAR, borderValue=float(S.ts[-1]))
    S.lay_t = cv2.GaussianBlur(tmap, (0, 0), 1.0)
    S.reveal_fade = 2.5 * S.speed / 6 * 1.0  # session seconds

    # ---- static texts ----
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    put_text(d, (56, S.hdr_y), "AGI House: iPhone LiDAR walkthrough", 38, 650)
    put_text(d, (58, S.hdr_y + 52), "RGB  ·  Depth  ·  Bird's-eye  ·  Path", 20, 450, TEXT2)
    # panel labels
    put_text(d, (x0 + 30, y0 + 28), "BIRD'S-EYE VIEW", 15, 650, TEXT2)
    put_text(d, (x0 + 30, y0 + 48), "floor plan from the fused LiDAR scan", 14, 400, TEXT3)
    # legend: path gradient
    lx1 = x1 - 30
    put_text(d, (lx1, y0 + 30), "now", 14, 500, TEXT2, anchor="ra")
    lw = 120
    tw = d.textlength("now", font=font(14, 500))
    gx1 = lx1 - tw - 10
    gx0 = gx1 - lw
    put_text(d, (gx0 - 10, y0 + 30), "start", 14, 500, TEXT2, anchor="ra")
    arr = np.asarray(img).astype(np.float32)
    for i in range(lw):
        blend(arr, grad_color(i / (lw - 1)), np.ones((6, 1), np.float32), int(gx0) + i, y0 + 36)
    # rounded ends: cheap dots
    for cx, u in ((gx0, 0), (gx0 + lw - 1, 1)):
        dot = disc(3.0, cx, y0 + 39, 8)
        blend(arr, grad_color(u), dot[0], dot[1], dot[2])
    # scale bar 5 m
    sb_len = 5 * S.view.s
    sbx, sby = x0 + 30, y1 - 34
    blend(arr, np.array(TEXT2, np.float32), np.ones((2, int(sb_len)), np.float32) * 0.8, sbx, sby)
    blend(arr, np.array(TEXT2, np.float32), np.ones((9, 2), np.float32) * 0.8, sbx, sby - 7)
    blend(arr, np.array(TEXT2, np.float32), np.ones((9, 2), np.float32) * 0.8, sbx + int(sb_len) - 2, sby - 7)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    put_text(d, (sbx + sb_len + 10, sby + 1), "5 m", 14, 500, TEXT2, anchor="lm")
    # image panel labels drawn per-frame (on top of imagery)
    # stats panel headings
    sx0, sy0, sx1, sy1 = S.stat_rect
    put_text(d, (sx0 + 28, sy0 + 26), "LIVE CAPTURE", 15, 650, TEXT2)
    cw = (sx1 - sx0 - 56) / 3
    S.stat_cols = [sx0 + 28 + i * cw for i in range(3)]
    for i, lab in enumerate(["Floor observed", "Walking pace", "People masked"]):
        put_text(d, (S.stat_cols[i], sy0 + 66), lab, 16, 450, TEXT2)
    # header right: device chip
    S.static = np.asarray(img).astype(np.uint8).copy()
    S.static_f = None


def disc(r, cx, cy, size=None):
    """Anti-aliased disc alpha at subpixel centre; returns (alpha, x, y)."""
    size = size or int(math.ceil(r * 2 + 4))
    x0, y0 = int(math.floor(cx - size / 2)), int(math.floor(cy - size / 2))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx + x0 + 0.5 - cx, yy + y0 + 0.5 - cy)
    return np.clip(r - d + 0.5, 0, 1), x0, y0


def chip(txt, size=15, weight=600, fg=TEXT, bg=(255, 255, 255, 22), dot=None, padx=12, pady=7):
    """Rounded pill RGBA sprite (float, straight alpha) rendered at 3x for clean edges."""
    ss = 3
    f = font(size * ss, weight)
    bb = f.getbbox(txt)
    asc, desc = f.getmetrics()
    th = int(size * ss * 0.72)
    dw = int(size * ss * 0.95) if dot is not None else 0
    W = bb[2] - bb[0] + 2 * padx * ss + dw
    H = th + 2 * pady * ss
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, W - 1, H - 1], H // 2, fill=tuple(bg))
    if dot is not None:
        r = int(size * ss * 0.24)
        cx, cy = padx * ss + r, H // 2
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(c) for c in dot) + (255,))
    dr.text((padx * ss + dw - bb[0], H // 2), txt, font=f, fill=tuple(fg) + (255,), anchor="lm")
    im = im.resize((round(W / ss), round(H / ss)), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    return a


def comp_sprite(dst, spr, x, y, alpha=1.0):
    blend(dst, spr[..., :3] * 255.0, spr[..., 3] * alpha, int(round(x)), int(round(y)))


# ---------------- per-frame rendering ----------------

_G: Scene | None = None
_cache = {}


def kf_index(S, t):
    return int(np.clip(np.searchsorted(S.ts, t, "right") - 1, 0, S.n - 1))


def load_rgb(S, i, size):
    key = ("rgb", i, size)
    if key in _cache:
        return _cache[key]
    im = cv2.imread(os.path.join(S.fr, S.stems[i] + ".jpg"), cv2.IMREAD_COLOR)[..., ::-1]
    im = np.ascontiguousarray(np.rot90(im, S.rot_k))
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    _cache.clear() if len(_cache) > 16 else None
    _cache[key] = im
    return im


def load_depth(S, i, size):
    key = ("dep", i, size)
    if key in _cache:
        return _cache[key]
    d = cv2.imread(os.path.join(S.fr, S.stems[i] + "_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    mk = cv2.imread(os.path.join(S.fr, S.stems[i] + "_mask.png"), cv2.IMREAD_GRAYSCALE)
    d = np.ascontiguousarray(np.rot90(d, S.rot_k))
    valid = (d > 0.05).astype(np.float32)
    near, far = 0.3, 5.0
    u = np.clip((np.log(np.maximum(d, near)) - math.log(near)) / (math.log(far) - math.log(near)), 0, 1)
    u = cv2.resize(u, size, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(valid, size, interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap(np.clip((1 - u) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    col = col.astype(np.float32) * 0.92
    out = col * v[..., None] + np.array([14, 16, 22], np.float32) * (1 - v[..., None])
    if mk is not None:
        m = np.ascontiguousarray(np.rot90(mk, S.rot_k)).astype(np.float32) / 255
        m = cv2.GaussianBlur(cv2.resize(m, size, interpolation=cv2.INTER_LINEAR), (0, 0), 1.2)
        m = smoothstep(0.3, 0.7, m)
        edge = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
        out += (MAGENTA - out) * (m * 0.5)[..., None]
        out += (np.array([255, 170, 225], np.float32) - out) * (edge * 0.9)[..., None]
    _cache[key] = out
    return out


def person_level(S, t):
    """Smoothed 0..1 'person present' signal around session time t."""
    w = 1.2 * S.speed / 6
    sel = np.abs(S.ts - t) < w
    if not sel.any():
        return 0.0
    return float(smoothstep(0.15, 0.5, S.person_on[sel].mean()))


def draw_map(S, t, canvas, show_dot=True, reveal_all=False):
    """canvas: float32 (mh, mw, 3) map-panel crop. t: session seconds."""
    mh, mw = canvas.shape[:2]
    # coverage reveal
    if reveal_all:
        rev = np.ones((mh, mw), np.float32)
    else:
        rev = np.clip((t - S.lay_t) / S.reveal_fade + 1.0, 0, 1)
        rev = rev * rev * (3 - 2 * rev)
    a = S.lay_a * rev
    canvas *= (1 - S.lay_shadow * rev * 0.8)[..., None]
    canvas += (np.array([255, 255, 255], np.float32) - canvas) * (S.lay_ghost * (1 - rev))[..., None]
    canvas += (S.lay_col - canvas) * a[..., None]

    # path
    ss = 2
    k = int(np.searchsorted(S.dense_t, t, "right"))
    pts = S.dense[:k]
    cur = S.pos_f(min(t, S.session_dur))
    pts = np.vstack([pts, cur[None]]) if len(pts) else cur[None]
    if len(pts) >= 2:
        P = S.view(pts) * ss * 4  # shift=2 subpixel
        P = np.round(P).astype(np.int32)
        shadow = np.zeros((mh * ss, mw * ss), np.uint8)
        cv2.polylines(shadow, [P.reshape(-1, 1, 2)], False, 255, 16, cv2.LINE_AA, shift=2)
        lay = np.zeros((mh * ss, mw * ss, 4), np.uint8)
        nseg = len(P) - 1
        step = 6
        tt = np.concatenate([S.dense_t[:k], [t]])
        for s0 in range(0, nseg, step):
            s1 = min(nseg, s0 + step)
            c = grad_color(tt[(s0 + s1) // 2] / S.session_dur)
            cv2.polylines(lay, [P[s0:s1 + 1].reshape(-1, 1, 2)], False,
                          (float(c[0]), float(c[1]), float(c[2]), 255.0), 7, cv2.LINE_AA, shift=2)
        sh = cv2.resize(shadow, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        sh = cv2.GaussianBlur(sh, (0, 0), 2.0)
        canvas *= (1 - 0.28 * sh)[..., None]
        L1 = cv2.resize(lay, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32)
        pa = L1[..., 3:4] / 255.0
        canvas[:] = canvas * (1 - pa) + L1[..., :3]

    # landmarks
    for lm in S.landmarks:
        if t < lm["t"] and not reveal_all:
            continue
        age = 99 if reveal_all else (t - lm["t"]) / S.speed  # video seconds
        sc = float(ease_out_back(age / 0.45))
        if sc <= 0.01:
            continue
        draw_pin(canvas, S.view(lm["xy"]), sc, lm, min(1.0, age / 0.3))

    if not show_dot:
        return
    # heading wedge
    tc = min(t, S.session_dur)
    p = S.view(cur)
    yaw = float(S.yaw_f(tc))
    dvec = lambda ang: S.view(cur + 2.2 * np.array([math.cos(ang), math.sin(ang)])) - p
    half = S.hfov / 2
    Lpx = 2.2 * S.view.s
    R = int(Lpx + 4)
    x0, y0 = int(p[0]) - R, int(p[1]) - R
    size = 2 * R
    angs = np.linspace(yaw - half, yaw + half, 24)
    poly = [p] + [p + dvec(a) for a in angs]
    poly = (np.array(poly) - [x0, y0]) * 2 * 4
    m = np.zeros((size * 2, size * 2), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 255, cv2.LINE_AA, shift=2)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx + x0 + 0.5 - p[0], yy + y0 + 0.5 - p[1]) / Lpx
    fall = np.clip(1 - rr, 0, 1) ** 0.9
    cc = grad_color(tc / S.session_dur)
    blend(canvas, cc * 0.85, m * fall * 0.55, x0, y0)
    # glowing dot
    gs = 64
    gx0, gy0 = int(p[0]) - gs // 2, int(p[1]) - gs // 2
    yy, xx = np.mgrid[0:gs, 0:gs].astype(np.float32)
    d = np.hypot(xx + gx0 + 0.5 - p[0], yy + gy0 + 0.5 - p[1])
    pulse = 1 + 0.15 * math.sin(t / S.speed * 2 * math.pi * 0.8)
    blend(canvas, cc, np.exp(-(d / (11 * pulse)) ** 2) * 0.55, gx0, gy0)
    blend(canvas, np.array([255, 255, 255], np.float32), np.clip(8.5 - d + 0.5, 0, 1), gx0, gy0)
    blend(canvas, cc, np.clip(5.8 - d + 0.5, 0, 1), gx0, gy0)


_pin_cache = {}


def draw_pin(canvas, p, sc, lm, alpha):
    r = 7.5 * sc
    if r > 0.5:
        a, x, y = disc(r + 2.5, p[0], p[1])
        sh, _, _ = disc(r + 4, p[0], p[1] + 2, a.shape[0])
        blend(canvas, np.zeros(3, np.float32), sh * 0.25, x, y)
        blend(canvas, np.array([255, 255, 255], np.float32), a, x, y)
        a, x, y = disc(r, p[0], p[1], a.shape[0])
        blend(canvas, AMBER, a, x, y)
    if lm.get("lab") is None:
        return
    dx, dy = lm["lab"]
    comp_sprite(canvas, lm["spr"], p[0] + dx, p[1] + dy + 4 * (1 - min(1, sc)), alpha * min(1.0, sc))


def paste_panel(frame, img, rect, mask):
    x0, y0, x1, y1 = rect
    blend(frame, img, mask, x0, y0)


def render_main(S, t, frame, end_mode=False):
    """Live walkthrough layout at session time t into float frame (H, W, 3)."""
    x0, y0, x1, y1 = S.map_rect
    canvas = frame[y0:y1, x0:x1]
    draw_map(S, t, canvas, show_dot=not end_mode, reveal_all=end_mode)
    tc = min(t, S.session_dur)
    dist = float(np.interp(tc, S.dense_t, S.dense_dist))
    fi = kf_index(S, tc)
    pr = person_level(S, tc)
    # distance counter on the map
    if not end_mode:
        pw, ph = S.rgb_rect[2] - S.rgb_rect[0], S.rgb_rect[3] - S.rgb_rect[1]
        paste_panel(frame, load_rgb(S, fi, (pw, ph)), S.rgb_rect, S.rgb_mask)
        paste_panel(frame, load_depth(S, fi, (pw, ph)), S.dep_rect, S.dep_mask)
        # top gradient scrims for labels
        for rect in (S.rgb_rect, S.dep_rect):
            g = np.linspace(0.55, 0, 64, dtype=np.float32)[:, None] * np.ones((1, rect[2] - rect[0]), np.float32)
            g *= S.rgb_mask[:64]
            blend(frame, np.zeros(3, np.float32), g, rect[0], rect[1])
        if "rgb_chip" not in _pin_cache:
            _pin_cache["rgb_chip"] = chip("RGB", 13, 650, bg=(0, 0, 0, 120))
            _pin_cache["dep_chip"] = chip("DEPTH · LiDAR", 13, 650, bg=(0, 0, 0, 120))
            _pin_cache["person"] = chip("Person removed from map", 13, 650, fg=(255, 255, 255),
                                        bg=(200, 30, 140, 225), dot=(255, 255, 255))
            _pin_cache["near"] = chip("near", 12, 600, bg=(0, 0, 0, 120), padx=9, pady=5)
            _pin_cache["far"] = chip("far", 12, 600, bg=(0, 0, 0, 120), padx=9, pady=5)
        comp_sprite(frame, _pin_cache["rgb_chip"], S.rgb_rect[0] + 14, S.rgb_rect[1] + 14)
        comp_sprite(frame, _pin_cache["dep_chip"], S.dep_rect[0] + 14, S.dep_rect[1] + 14)
        # depth colour legend
        dx0, dy1 = S.dep_rect[0] + 14, S.dep_rect[3] - 14
        lw = 110
        cmap = cv2.applyColorMap(np.linspace(255, 0, lw).astype(np.uint8)[None], cv2.COLORMAP_TURBO)[0, :, ::-1]
        nc = _pin_cache["near"]
        bar_x = dx0 + nc.shape[1] + 8
        bar = np.repeat(cmap[None].astype(np.float32), 6, 0)
        comp_sprite(frame, nc, dx0, dy1 - nc.shape[0])
        blend(frame, bar, rounded_mask(lw, 6, 3), bar_x, dy1 - nc.shape[0] // 2 - 3)
        comp_sprite(frame, _pin_cache["far"], bar_x + lw + 8, dy1 - nc.shape[0])
        if pr > 0.01:
            spr = _pin_cache["person"]
            comp_sprite(frame, spr, S.dep_rect[2] - 14 - spr.shape[1], S.dep_rect[1] + 14 - 6 * (1 - pr), pr)
    return dist, fi, pr


def draw_hud(S, img, t, dist, fi, pr, vt, end_mode=False):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = S.map_rect
    if not end_mode:
        put_text(d, (x1 - 30, y1 - 22), f"{dist:5.1f} m", 34, 650, TEXT, anchor="rs")
        put_text(d, (x1 - 30, y1 - 64), "DISTANCE WALKED", 12, 650, TEXT2, anchor="rs")
    # stats panel
    sx0, sy0, sx1, sy1 = S.stat_rect
    tc = min(t, S.session_dur)
    obs_area = float(((S.first <= tc) & S.free).sum() * S.res ** 2)
    i0 = max(0, kf_index(S, tc - 4))
    i1 = kf_index(S, tc)
    pace = 0.0
    if S.ts[i1] - S.ts[i0] > 0.5:
        pace = (np.interp(S.ts[i1], S.dense_t, S.dense_dist) - np.interp(S.ts[i0], S.dense_t, S.dense_dist)) / (
            S.ts[i1] - S.ts[i0])
    nper = int(S.person_on[:fi + 1].sum())
    vals = [(f"{obs_area:,.0f}", " m²"), (f"{pace:.2f}", " m/s"), (f"{nper}", " frames")]
    if end_mode:
        pass
    for i, (v, u) in enumerate(vals):
        x = S.stat_cols[i]
        f = font(44, 650)
        put_text(d, (x, sy0 + 140), v, 44, 650, TEXT, anchor="ls")
        w = d.textlength(v, font=f)
        put_text(d, (x + w + 2, sy0 + 140), u, 20, 500, TEXT2, anchor="ls")
    # coverage progress bar
    by = sy1 - 56
    put_text(d, (sx0 + 28, by - 14), "Walkthrough progress", 14, 500, TEXT2, anchor="ls")
    put_text(d, (sx1 - 28, by - 14), f"keyframe {fi + 1:,} / {S.n:,}", 14, 500, TEXT2, anchor="rs")
    # footer
    fy = S.foot_y
    put_text(d, (56, fy), f"{fmt_time(tc)}", 20, 600, TEXT, anchor="lm")
    w = d.textlength(fmt_time(tc), font=font(20, 600))
    put_text(d, (56 + w + 6, fy), f"/ {fmt_time(S.session_dur)}  elapsed", 20, 400, TEXT2, anchor="lm")
    items = [("Keyframe", f"{fi + 1:,} / {S.n:,}"), ("Distance", f"{dist:.1f} m"), ("Playback", f"×{S.speed} speed")]
    x = 420
    for lab, val in items:
        put_text(d, (x, fy), lab, 18, 450, TEXT2, anchor="lm")
        lw_ = d.textlength(lab, font=font(18, 450))
        put_text(d, (x + lw_ + 10, fy), val, 18, 600, TEXT, anchor="lm")
        x += lw_ + 10 + d.textlength(val, font=font(18, 600)) + 48
    return obs_area


def render_frame(vi: int) -> bytes:
    S = _G
    vt = vi / FPS
    out = render_np(S, vt)
    return out.tobytes()


def render_np(S, vt):
    if vt < S.t_title:
        # dip through the background: title out, then the dashboard in
        out_a = float(smoothstep(S.t_title - 0.9, S.t_title - 0.5, vt))
        in_a = float(smoothstep(S.t_title - 0.45, S.t_title, vt))
        bg = _pin_cache.get("title_bg")
        title = render_title(S, vt).astype(np.float32)
        if out_a <= 0:
            return title.astype(np.uint8)
        bg = _pin_cache["title_bg"]
        f = title * (1 - out_a) + bg * out_a
        if in_a > 0:
            f = f * (1 - in_a) + render_live(S, 0.0, vt).astype(np.float32) * in_a
        return np.clip(f, 0, 255).astype(np.uint8)
    if vt < S.t_title + S.main_dur:
        return render_live(S, (vt - S.t_title) * S.speed, vt)
    te = vt - S.t_title - S.main_dur
    end = render_end(S, te)
    fade = smoothstep(0, 0.7, te)
    if fade >= 1:
        return end
    main = render_live(S, S.session_dur, vt)
    return (main * (1 - fade) + end * fade).astype(np.uint8)


def render_live(S, t, vt):
    frame = S.static.astype(np.float32)
    dist, fi, pr = render_main(S, t, frame)
    # progress bar in stats panel and footer chips (numpy, for AA)
    sx0, sy0, sx1, sy1 = S.stat_rect
    by = sy1 - 50
    bw = sx1 - sx0 - 56
    blend(frame, np.array([255, 255, 255], np.float32), rounded_mask(bw, 8, 4) * 0.08, sx0 + 28, by)
    fw = max(8, int(bw * min(1, t / S.session_dur)))
    grad = np.stack([grad_color(u) for u in np.linspace(0, min(1, t / S.session_dur), fw)])[None].repeat(8, 0)
    blend(frame, grad, rounded_mask(fw, 8, 4), sx0 + 28, by)
    header_chips(S, frame)
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    draw_hud(S, img, t, dist, fi, pr, vt)
    return np.asarray(img)


def header_chips(S, frame, show_speed=True):
    if "hdr" not in _pin_cache:
        _pin_cache["hdr"] = [
            chip("100% on-device", 15, 600, fg=(220, 255, 240), bg=(16, 120, 90, 150), dot=GREEN),
            chip(f"×{S.speed} speed", 15, 600, bg=(255, 255, 255, 26)),
            chip(f"{S.n:,} RGB-D keyframes", 15, 600, bg=(255, 255, 255, 26)),
        ]
    x = W_OUT - 56
    for spr in _pin_cache["hdr"]:
        x -= spr.shape[1]
        comp_sprite(frame, spr, x, 62 - spr.shape[0] / 2)
        x -= 10
    # footer chip
    if "foot" not in _pin_cache:
        _pin_cache["foot"] = chip("iPhone LiDAR  ·  ARKit poses  ·  processed on-device", 14, 500, fg=TEXT2,
                                  bg=(255, 255, 255, 14))
    spr = _pin_cache["foot"]
    comp_sprite(frame, spr, W_OUT - 56 - spr.shape[1], S.foot_y - spr.shape[0] / 2)


def render_title(S, vt):
    if "title_bg" not in _pin_cache:
        bg = np.zeros((H_OUT, W_OUT, 3), np.float32) + BG
        yy, xx = np.mgrid[0:H_OUT, 0:W_OUT].astype(np.float32)
        bg += np.exp(-(((xx - 960) / 900) ** 2 + ((yy - 300) / 600) ** 2))[..., None] * np.array([18, 20, 44], np.float32)
        bg += np.exp(-(((xx - 1700) / 600) ** 2 + ((yy - 1100) / 500) ** 2))[..., None] * np.array([30, 10, 36], np.float32)
        # faint full map + path in the background
        x0, y0, x1, y1 = S.map_rect
        canvas = np.zeros((S.mh, S.mw, 3), np.float32)
        a = S.lay_a
        canvas += S.lay_col * a[..., None]
        _pin_cache["title_bg"] = bg
        _pin_cache["title_map"] = (canvas, a)
    frame = _pin_cache["title_bg"].copy()
    canvas, a = _pin_cache["title_map"]
    cx = (W_OUT - S.mw) // 2
    cy = (H_OUT - S.mh) // 2 + 20
    blend(frame, canvas, a * 0.06 * smoothstep(0, 1.5, vt), cx, cy)
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).convert("RGBA")
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)

    def fade_in(start, dur=0.7):
        u = float(smoothstep(start, start + dur, vt))
        return u, (1 - u) * 18

    u, dy = fade_in(0.1)
    d.text((960, 400 + dy), "AGI House", font=font(112, 700), fill=TEXT + (int(255 * u),), anchor="ms")
    u, dy = fade_in(0.45)
    d.text((960, 470 + dy), "From an iPhone LiDAR walkthrough to a robot-ready digital twin",
           font=font(34, 450), fill=TEXT2 + (int(255 * u),), anchor="ms")
    u, dy = fade_in(0.8)
    d.text((960, 540 + dy), "so a humanoid robot can escort elderly guests around the house",
           font=font(24, 400), fill=TEXT3 + (int(255 * u),), anchor="ms")
    img = Image.alpha_composite(img, lay).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    # chips row
    u, dy = fade_in(1.1)
    if "title_chips" not in _pin_cache:
        _pin_cache["title_chips"] = [
            chip(f"{S.n:,} RGB-D keyframes", 18, 600, bg=(255, 255, 255, 24), padx=16, pady=9),
            chip(f"{fmt_time(S.session_dur)} walk  ·  {S.path_len:.0f} m", 18, 600, bg=(255, 255, 255, 24), padx=16, pady=9),
            chip("100% on-device", 18, 600, fg=(220, 255, 240), bg=(16, 120, 90, 160), dot=GREEN, padx=16, pady=9),
        ]
    ch = _pin_cache["title_chips"]
    tw = sum(c.shape[1] for c in ch) + 14 * (len(ch) - 1)
    x = (W_OUT - tw) / 2
    for c in ch:
        comp_sprite(arr, c, x, 610 + dy, u)
        x += c.shape[1] + 14
    # gradient accent line
    u2 = float(smoothstep(0.3, 1.4, vt))
    lw = int(420 * u2)
    if lw > 2:
        g = np.stack([grad_color(v) for v in np.linspace(0, 1, lw)])[None].repeat(4, 0)
        blend(arr, g, rounded_mask(lw, 4, 2), 960 - lw // 2, 262)
    return arr.astype(np.uint8)


def render_end(S, te):
    frame = S.static.astype(np.float32)
    # replace right-side panels with a summary
    x0, y0, x1, y1 = S.map_rect
    rx0, rx1 = S.rgb_rect[0], S.dep_rect[2]
    # wipe right column back to background, then draw summary cards
    bgc = _pin_cache.get("end_bg")
    if bgc is None:
        bg = np.zeros((H_OUT, W_OUT, 3), np.float32) + BG
        yy, xx = np.mgrid[0:H_OUT, 0:W_OUT].astype(np.float32)
        bg += np.exp(-(((xx - 520) / 900) ** 2 + ((yy + 200) / 700) ** 2))[..., None] * np.array([14, 18, 34], np.float32)
        bg += np.exp(-(((xx - 1800) / 700) ** 2 + ((yy - 1200) / 600) ** 2))[..., None] * np.array([22, 10, 30], np.float32)
        bg[:, :rx0 - 20] = S.static[:, :rx0 - 20]
        bg[:S.top - 10] = S.static[:S.top - 10]
        # 2x2 cards
        cw = (rx1 - rx0 - 20) // 2
        chh = 230
        cards = []
        for i in range(4):
            cx = rx0 + (i % 2) * (cw + 20)
            cy = S.top + (i // 2) * (chh + 20)
            m = rounded_mask(cw, chh, 22)
            sh = np.zeros((chh + 80, cw + 80), np.float32)
            sh[40:40 + chh, 40:40 + cw] = m
            blend(bg, np.zeros(3, np.float32), cv2.GaussianBlur(sh, (0, 0), 16) * 0.55, cx - 40, cy - 30)
            blend(bg, np.array(PANEL, np.float32), m, cx, cy)
            cards.append((cx, cy, cw, chh))
        _pin_cache["end_bg"] = bgc = bg
        _pin_cache["end_cards"] = cards
    frame = bgc.copy()
    draw_map(S, S.session_dur, frame[y0:y1, x0:x1], show_dot=False, reveal_all=True)
    # start / end markers
    canvas = frame[y0:y1, x0:x1]
    for xy, col in ((S.dense[0], CYAN), (S.dense[-1], PINK)):
        p = S.view(xy)
        a, x, y = disc(9, p[0], p[1])
        blend(canvas, np.array([255, 255, 255], np.float32), a, x, y)
        a, x, y = disc(6, p[0], p[1], a.shape[0])
        blend(canvas, col, a, x, y)
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).convert("RGBA")
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    stats = [
        (f"{S.area_free:,.0f}", "m²", "walkable floor mapped"),
        (f"{S.path_len:,.0f}", "m", "path walked"),
        (f"{S.n:,}", "", "RGB-D keyframes"),
        (fmt_time(S.session_dur), "min", "capture duration"),
    ]
    for i, ((cx, cy, cw, chh), (v, unit, lab)) in enumerate(zip(_pin_cache["end_cards"], stats)):
        u = float(smoothstep(0.15 + 0.12 * i, 0.75 + 0.12 * i, te))
        a = int(255 * u)
        dy = (1 - u) * 14
        f = font(72, 700)
        d.text((cx + 32, cy + 130 + dy), v, font=f, fill=TEXT + (a,), anchor="ls")
        w = d.textlength(v, font=f)
        if unit:
            d.text((cx + 32 + w + 8, cy + 130 + dy), unit, font=font(28, 500), fill=TEXT2 + (a,), anchor="ls")
        d.text((cx + 32, cy + 176 + dy), lab, font=font(22, 450), fill=TEXT2 + (a,), anchor="ls")
    u = float(smoothstep(0.7, 1.4, te))
    ly = S.top + 2 * 250 + 40
    d.text((rx0 + 4, ly + 30), "Ready for the robot.", font=font(40, 700), fill=TEXT + (int(255 * u),), anchor="ls")
    d.text((rx0 + 4, ly + 76), "Clean floor plan, people removed, landmarks named:", font=font(22, 400),
           fill=TEXT2 + (int(255 * u),), anchor="ls")
    d.text((rx0 + 4, ly + 108), "the humanoid can now plan escort routes through AGI House.", font=font(22, 400),
           fill=TEXT2 + (int(255 * u),), anchor="ls")
    names = sorted({lm["name"] for lm in S.landmarks})
    if names:
        d.text((rx0 + 4, ly + 170), f"{len(S.landmarks)} LANDMARKS NAMED ON THE MAP", font=font(15, 650),
               fill=TEXT2 + (int(255 * u),), anchor="ls")
    img = Image.alpha_composite(img, lay).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    if names:
        if "end_names" not in _pin_cache:
            _pin_cache["end_names"] = [chip(n, 16, 600, bg=(255, 255, 255, 22), dot=tuple(AMBER), padx=12, pady=7)
                                       for n in names]
        x, y = rx0, ly + 190
        for i, spr in enumerate(_pin_cache["end_names"]):
            if x + spr.shape[1] > rx1:
                x, y = rx0, y + spr.shape[0] + 10
            if y + spr.shape[0] > S.bot:
                break
            uu = float(smoothstep(0.9 + 0.05 * i, 1.3 + 0.05 * i, te))
            comp_sprite(arr, spr, x, y, uu)
            x += spr.shape[1] + 10
    # legend in map: start/end
    if "end_leg" not in _pin_cache:
        _pin_cache["end_leg"] = (chip("start", 13, 600, bg=(0, 0, 0, 0), dot=tuple(CYAN), padx=4),
                                 chip("end", 13, 600, bg=(0, 0, 0, 0), dot=tuple(PINK), padx=4))
    s1, s2 = _pin_cache["end_leg"]
    comp_sprite(arr, s1, x1 - 30 - s2.shape[1] - 12 - s1.shape[1], y1 - 30 - s1.shape[0] / 2 - 4)
    comp_sprite(arr, s2, x1 - 30 - s2.shape[1], y1 - 30 - s2.shape[0] / 2 - 4)
    header_chips(S, arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------- encoding ----------------

def _init_worker():
    cv2.setNumThreads(1)


def ffmpeg_writer(path, w=W_OUT, h=H_OUT, fps=FPS, crf=16):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_walkthrough(S, args):
    global _G
    _G = S
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "walkthrough_1080p.mp4")
    stills_dir = os.path.join(args.out, "stills")
    os.makedirs(stills_dir, exist_ok=True)
    N = S.nframes
    t_main0 = S.t_title
    still_at = {
        int(1.6 * FPS): "01_title.png",
        int((t_main0 + 0.30 * S.main_dur) * FPS): "02_walkthrough_30pct.png",
        int((t_main0 + 0.62 * S.main_dur) * FPS): "03_walkthrough_60pct.png",
        int((t_main0 + 0.93 * S.main_dur) * FPS): "04_walkthrough_90pct.png",
        N - 3: "05_summary.png",
    }
    # a still with a person in view, if any
    pidx = np.nonzero(S.person_on)[0]
    if len(pidx):
        # pick the keyframe with the most person pixels in the middle 80%
        mid = pidx[(S.ts[pidx] > 0.1 * S.session_dur) & (S.ts[pidx] < 0.9 * S.session_dur)]
        if len(mid):
            best = mid[np.argmax(S.person[mid])]
            still_at[int((t_main0 + S.ts[best] / S.speed) * FPS) + 1] = "06_person_removed.png"
    hl0 = int((t_main0 + 0.35 * S.main_dur) * FPS)
    wr = ffmpeg_writer(out)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for vi, buf in enumerate(pool.imap(render_frame, range(N), chunksize=4)):
            wr.stdin.write(buf)
            if vi in still_at:
                Image.frombytes("RGB", (W_OUT, H_OUT), buf).save(os.path.join(stills_dir, still_at[vi]))
            if vi % 300 == 0:
                print(f"  frame {vi}/{N}  {time.time() - t0:.0f} s", flush=True)
    wr.stdin.close()
    wr.wait()
    print(f"wrote {out} ({N / FPS:.1f} s) in {time.time() - t0:.0f} s")
    # highlight: 10 s cut + gif
    hl = os.path.join(args.out, "highlight_10s.mp4")
    ss = hl0 / FPS
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{ss:.3f}", "-i", out, "-t", "10",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", hl], check=True)
    gif = os.path.join(args.out, "highlight_10s.gif")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{ss:.3f}", "-t", "10", "-i", out, "-vf",
                    "fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];"
                    "[b][p]paletteuse=dither=sierra2_4a", gif], check=True)
    return out


# ---------------- video 2: orbit around the point cloud ----------------

def render_orbit(S, args):
    import open3d as o3d
    ply = args.scan
    mesh = o3d.io.read_triangle_mesh(ply)
    if len(mesh.triangles):
        pcd = mesh.sample_points_uniformly(args.orbit_points)
        V = np.asarray(pcd.points)
        C = np.asarray(pcd.colors)
    else:
        pc = o3d.io.read_point_cloud(ply)
        V, C = np.asarray(pc.points), np.asarray(pc.colors)
    if "fused" in os.path.basename(ply):  # ingest frame -> map frame
        V = V @ S.L[:3, :3].T + S.L[:3, 3]
    keep = (V[:, 2] > -0.1) & (V[:, 2] < 2.0)
    V, C = V[keep].astype(np.float32), C[keep].astype(np.float32)
    # rotate into the axis-aligned map frame used by the 2D views
    c, s = math.cos(-S.view.phi), math.sin(-S.view.phi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    ctr = np.array([S.view.ctr[0], S.view.ctr[1], 0], np.float32)
    V = (V - ctr) @ Rz.T
    path = np.c_[S.dense, np.full(len(S.dense), 0.06)].astype(np.float32)
    path = (path - ctr) @ Rz.T
    path_u = S.dense_t / S.session_dur
    cols = (C * 255).astype(np.float32)
    # colour boost
    cols = np.clip((cols - 128) * 1.1 + 132, 0, 255)
    extent = np.percentile(np.abs(V[:, :2]), 99, axis=0)
    radius = float(np.hypot(*extent)) * 1.25
    dur = args.orbit_secs
    N = int(dur * FPS)
    out = os.path.join(args.out, "twin_orbit_1080p.mp4")
    preview = getattr(args, "orbit_preview", False)
    wr = None if preview else ffmpeg_writer(out)
    W, H = W_OUT, H_OUT
    bg = np.zeros((H, W, 3), np.float32) + BG
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg += np.exp(-(((xx - 960) / 1000) ** 2 + ((yy - 200) / 700) ** 2))[..., None] * np.array([18, 22, 44], np.float32)
    FOV = 48
    scrim = np.clip(1 - yy / 170, 0, 1)[..., None] ** 1.5 * 0.75
    t0 = time.time()
    frames = [0, N // 2, N - 16] if preview else range(N)
    for fi in frames:
        u = fi / (N - 1)
        e = 0.5 - 0.5 * math.cos(math.pi * u)
        az = math.radians(-60 + 140 * e)
        el = math.radians(58 - 18 * e)
        dist = radius * (2.45 - 0.4 * e)
        eye = np.array([dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az), dist * math.sin(el)])
        # aim slightly toward the camera so the near edge of the house is not clipped
        target = np.array([0.07 * radius * math.cos(az), 0.07 * radius * math.sin(az), 0.3])
        img = splat(V, cols, eye, target, W, H, fov=FOV, bg=bg)
        # path: draw with growth over the first 70% of the clip
        grow = min(1.0, u / 0.7)
        k = max(2, int(len(path) * grow))
        P, z = project(path[:k], eye, target, W, H, FOV)
        frame = img
        lay = np.zeros((H * 2, W * 2, 4), np.uint8)
        Pi = np.round(P * 2 * 4).astype(np.int32)
        step = 8
        for s0 in range(0, k - 1, step):
            s1 = min(k - 1, s0 + step)
            cc = grad_color(path_u[(s0 + s1) // 2])
            cv2.polylines(lay, [Pi[s0:s1 + 1].reshape(-1, 1, 2)], False,
                          (float(cc[0]), float(cc[1]), float(cc[2]), 255.0), 8, cv2.LINE_AA, shift=2)
        L1 = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        glow = cv2.GaussianBlur(L1, (0, 0), 6)
        frame = frame + glow[..., :3] * 0.6
        pa = L1[..., 3:4] / 255
        frame = frame * (1 - pa) + L1[..., :3]
        # head dot
        p = P[-1]
        cc = grad_color(path_u[k - 1])
        a, x, y = disc(9, p[0], p[1], 80)
        g = np.hypot(*np.mgrid[0:80, 0:80].astype(np.float32) - 40 + 0.5)
        blend(frame, cc, np.exp(-(g / 16) ** 2) * 0.6, x, y)
        blend(frame, np.array([255, 255, 255], np.float32), a, x, y)
        a, x, y = disc(6, p[0], p[1], 80)
        blend(frame, cc, a, x, y)
        frame = frame * (1 - scrim) + BG * scrim
        im = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        put_text(d, (56, 40), "AGI House: digital twin", 38, 650)
        put_text(d, (58, 92), f"Fused LiDAR scan · {len(V) / 1e6:.1f} M points · walkthrough path in 3D", 20, 450, TEXT2)
        arr = np.asarray(im).astype(np.float32)
        if "orb" not in _pin_cache:
            _pin_cache["orb"] = chip("100% on-device", 15, 600, fg=(220, 255, 240), bg=(16, 120, 90, 150), dot=GREEN)
        spr = _pin_cache["orb"]
        comp_sprite(arr, spr, W - 56 - spr.shape[1], 62 - spr.shape[0] / 2)
        fade = min(1.0, fi / 15, (N - 1 - fi) / 15)
        arr = bg + (arr - bg) * fade
        out_b = np.clip(arr, 0, 255).astype(np.uint8)
        if preview:
            os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)
            Image.fromarray(out_b).save(os.path.join(args.out, "preview", f"orbit_{fi:04d}.png"))
            print("orbit frame", fi, f"{time.time() - t0:.1f} s")
            continue
        wr.stdin.write(out_b.tobytes())
        if fi == int(N * 0.75):
            Image.fromarray(out_b).save(os.path.join(args.out, "stills", "07_twin_orbit.png"))
        if fi % 60 == 0:
            print(f"  orbit {fi}/{N} {time.time() - t0:.0f} s", flush=True)
    if preview:
        return
    wr.stdin.close()
    wr.wait()
    print(f"wrote {out} ({dur:.1f} s)")


def look_at(eye, target):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f]).astype(np.float32)  # rows: cam x (right), y (down), z (forward)


def project(P, eye, target, W, H, fov):
    R = look_at(eye, target)
    Pc = (P - eye.astype(np.float32)) @ R.T
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    z = np.maximum(Pc[:, 2], 1e-3)
    return np.stack([W / 2 + f * Pc[:, 0] / z, H / 2 + f * Pc[:, 1] / z], 1), z


def splat(V, cols, eye, target, W, H, fov, bg=None):
    """Z-buffered point splatting with 2x2 footprint and simple depth shading, rendered at 2x."""
    ss = 2
    W2, H2 = W * ss, H * ss
    P, z = project(V, eye, target, W2, H2, fov)
    ok = (z > 0.1) & (P[:, 0] >= 0) & (P[:, 0] < W2 - 2) & (P[:, 1] >= 0) & (P[:, 1] < H2 - 2)
    P, z, c = P[ok], z[ok], cols[ok]
    order = np.argsort(-z)  # far first, near overwrite
    P, z, c = P[order], z[order], c[order]
    xi = P[:, 0].astype(np.int32)
    yi = P[:, 1].astype(np.int32)
    img = np.zeros((H2, W2, 3), np.float32)
    zb = np.full((H2, W2), np.inf, np.float32)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            img[yi + dy, xi + dx] = c
            zb[yi + dy, xi + dx] = z
    filled = np.isfinite(zb)
    # fill small holes: normalised box blur of the splats
    a0 = filled.astype(np.float32)
    num = cv2.blur(img * a0[..., None], (7, 7))
    den = cv2.blur(a0, (7, 7))
    hole = (~filled) & (den > 0.35)
    img[hole] = num[hole] / den[hole][:, None]
    zb[hole] = cv2.erode(np.where(filled, zb, np.float32(1e9)), np.ones((7, 7), np.uint8))[hole]
    filled = filled | hole
    # eye-dome-ish edge darkening from depth discontinuities
    zz = np.where(filled, zb, zb[filled].max() if filled.any() else 1)
    lz = np.log(zz)
    nb = np.maximum.reduce([np.roll(lz, s, a) - lz for s in (-3, 3) for a in (0, 1)])
    shade = np.exp(-np.clip(nb, 0, None) * 6)[..., None]
    img = img * (0.35 + 0.65 * shade)
    a = filled.astype(np.float32)[..., None]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)[..., None]
    if bg is None:
        bg = np.zeros((H, W, 3), np.float32) + BG
    return bg * (1 - a) + img


# ---------------- cli ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=os.path.join(HERE, "captures", "session_20261024-1347.zip"))
    ap.add_argument("--map", default=os.path.join(HERE, "maps", "agihouse"))
    ap.add_argument("--scan", default=os.path.join(HERE, "scans", "agihouse", "fused.ply"))
    ap.add_argument("--out", default=os.path.join(HERE, "videos"))
    ap.add_argument("--speed", type=int, default=0, help="playback speed-up (default: fit --target)")
    ap.add_argument("--target", type=float, default=72, help="target length of the live part (s)")
    ap.add_argument("--title-secs", type=float, default=3.0)
    ap.add_argument("--end-secs", type=float, default=3.5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--orbit", action="store_true", help="also render twin_orbit_1080p.mp4")
    ap.add_argument("--orbit-only", action="store_true")
    ap.add_argument("--orbit-preview", action="store_true", help="write 3 orbit PNGs only")
    ap.add_argument("--orbit-secs", type=float, default=12)
    ap.add_argument("--orbit-points", type=int, default=2_000_000)
    ap.add_argument("--preview", default="", help="comma list of video seconds: write PNGs only")
    args = ap.parse_args()
    S = build_scene(args)
    if args.preview:
        os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)
        for v in args.preview.split(","):
            vt = float(v)
            p = os.path.join(args.out, "preview", f"t{vt:06.2f}.png")
            Image.fromarray(render_np(S, vt)).save(p)
            print("wrote", p)
        return
    if args.orbit_preview:
        render_orbit(S, args)
        return
    if not args.orbit_only:
        render_walkthrough(S, args)
    if args.orbit or args.orbit_only:
        render_orbit(S, args)
    print(f"alignment: {S.on_free_pct:.1f}% of camera path keyframes on free cells; "
          f"{len(S.landmarks)} landmarks shown")


if __name__ == "__main__":
    main()
