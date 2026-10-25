"""Synthetic RGB-D capture session in the exact docs/rgbd_format.md layout.

Ray-casts a two-room house (walls, a doorway with a header, ceiling, kitchen counter,
couch), a moving "person" cylinder and a "bag" on the floor, along a phone walkthrough.
Matches the iOS app: depth PNG is uint16 millimetres (session.json still says
depth_units "meters"), mask 255 = person and conf 0/1/2 at depth resolution, pose is ARKit
camera.transform row-major. Poses are ARKit-style: world Y-up, origin at the first camera position, world -Z along
the first camera's horizontal heading; camera looks down its -Z, +Y up (OpenGL).

    .venv/bin/python tools/fake_session.py captures/session_fake            # folder
    .venv/bin/python tools/fake_session.py captures/session_fake --zip      # + .zip (top-level
                                                    session_fake/ folder, like the iOS app)
    .venv/bin/python tools/fake_session.py captures/x --mask-dropout 6       # person mask
                                                    missing on every 6th frame (carving test)

Also writes <session>/gt.json (not part of the format, ingest ignores it): the room ->
ARKit-world transform and ground-truth object positions in room coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

# Room frame: Z-up, metres, floor at z=0 (same layout as scan_map.synth + clutter).
WALL_H = 2.5
DOOR = dict(x=4.5, y0=2.2, y1=3.8, header=2.1)
COUNTER = dict(c=(7.6, 4.5), s=(0.6, 2.5), h=0.9)
COUCH = dict(c=(2.0, 5.3), s=(2.0, 0.9), h=0.8)
BAG = dict(c=(1.1, 1.3), s=(0.4, 0.3), h=0.3)
PERSON_R, PERSON_H = 0.35, 1.7

COLORS = {"floor": (150, 110, 75), "ceiling": (240, 240, 238), "wall": (215, 205, 185),
          "counter": (190, 190, 200), "couch": (40, 95, 60), "bag": (150, 25, 30),
          "person_legs": (40, 45, 70), "person_shirt": (60, 110, 200), "person_head": (225, 180, 150)}

N_FRAMES = 60
CAM_H = 1.35
PITCH = math.radians(-22)
DEPTH_W, DEPTH_H = 256, 192
RGB_W, RGB_H = 960, 720
RGB_FX = 740.0   # ~66 deg horizontal FOV, like the iPhone wide camera at 960 px


def box_mesh(cx, cy, sx, sy, h, z0=0.0):
    m = o3d.geometry.TriangleMesh.create_box(sx, sy, h)
    m.translate((cx - sx / 2, cy - sy / 2, z0))
    return m


def static_parts():
    parts = [("floor", box_mesh(4, 3, 8.2, 6.2, 0.02, -0.02)),
             ("ceiling", box_mesh(4, 3, 8.2, 6.2, 0.02, WALL_H)),
             ("wall", box_mesh(4, -0.05, 8.2, 0.1, WALL_H)), ("wall", box_mesh(4, 6.05, 8.2, 0.1, WALL_H)),
             ("wall", box_mesh(-0.05, 3, 0.1, 6.2, WALL_H)), ("wall", box_mesh(8.05, 3, 0.1, 6.2, WALL_H))]
    x, y0, y1, hd = DOOR["x"], DOOR["y0"], DOOR["y1"], DOOR["header"]
    parts += [("wall", box_mesh(x, y0 / 2, 0.1, y0, WALL_H)),
              ("wall", box_mesh(x, (y1 + 6) / 2, 0.1, 6 - y1, WALL_H)),
              ("wall", box_mesh(x, (y0 + y1) / 2, 0.1, y1 - y0, WALL_H - hd, hd))]   # door header
    parts += [("counter", box_mesh(*COUNTER["c"], *COUNTER["s"], COUNTER["h"])),
              ("couch", box_mesh(*COUCH["c"], *COUCH["s"], 0.45)),                    # seat
              ("couch", box_mesh(COUCH["c"][0], COUCH["c"][1] + 0.3, COUCH["s"][0], 0.3, COUCH["h"])),
              ("bag", box_mesh(*BAG["c"], *BAG["s"], BAG["h"]))]
    return parts


def person_parts(px, py):
    out = []
    for name, z0, z1 in (("person_legs", 0.0, 0.85), ("person_shirt", 0.85, 1.45), ("person_head", 1.45, PERSON_H)):
        r = PERSON_R if name != "person_head" else 0.14
        c = o3d.geometry.TriangleMesh.create_cylinder(r, z1 - z0, resolution=24)
        c.translate((px, py, (z0 + z1) / 2))
        out.append((name, c))
    return out


def lerp_path(pts, t):
    """Piecewise-linear point along a polyline, t in [0,1]."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = t * seg.sum()
    for i, L in enumerate(seg):
        if s <= L or i == len(seg) - 1:
            return pts[i] + (pts[i + 1] - pts[i]) * min(1.0, s / max(L, 1e-9))
        s -= L


def walkthrough():
    """(x, y, yaw) in the room frame for each keyframe: sweep living room, walk through
    the door, sweep the kitchen."""
    out = []
    for i in range(25):
        a = 2 * math.pi * i / 25
        out.append((2.2 + 0.6 * math.cos(a), 3.0 + 0.6 * math.sin(a), a))
    for i in range(10):
        out.append((3.0 + 3.2 * i / 9, 3.0, 0.0))
    for i in range(25):
        a = 2 * math.pi * i / 25
        out.append((6.3 + 0.5 * math.cos(a + math.pi), 3.0 + 0.5 * math.sin(a + math.pi), a))
    return out


def person_at(i, cam=None):
    """Someone walking laps of the living room, standing in the doorway while the camera
    heads for it, then walking along the kitchen. Moves ~0.4 m between keyframes."""
    if i <= 22:
        return tuple(lerp_path([(0.8, 4.2), (3.8, 4.2), (3.8, 0.8), (1.9, 0.7)], i / 22))
    if i <= 26:
        return 4.6, 3.0
    return tuple(lerp_path([(5.3, 0.8), (5.3, 5.3), (6.2, 5.3)], (i - 27) / 32))


def gl_cam_pose(x, y, yaw, h=CAM_H, pitch=PITCH):
    """Camera-to-room 4x4 in OpenGL camera convention (looks -Z, +Y up), room Z-up."""
    fwd = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, -fwd, (x, y, h)
    return T


def room_to_arkit(x0, y0, yaw0, h0=CAM_H):
    """ARKit world: origin at the first camera, Y up, -Z = first camera's heading."""
    fwd = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    up = np.array([0, 0, 1.0])
    ez = -fwd
    ex = np.cross(up, ez)
    R = np.stack([ex, up, ez])        # rows = arkit axes expressed in room coords
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([x0, y0, h0])
    return T


def render(scene, ids, T_cam_world, fx, fy, cx, cy, w, h):
    """Ray-cast in the ARKit world with a GL camera. Returns z-depth (m), object ids, normals."""
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))  # pixel-index convention, as open3d
    d_cam = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], -1)       # GL camera rays
    d_w = d_cam @ T_cam_world[:3, :3].T
    o_w = np.broadcast_to(T_cam_world[:3, 3], d_w.shape)
    rays = o3d.core.Tensor(np.concatenate([o_w, d_w], -1).astype(np.float32))
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()                        # along d_cam, whose camera-Z component is -1
    gid = ans["geometry_ids"].numpy().astype(np.int64)
    nrm = ans["primitive_normals"].numpy()
    hit = np.isfinite(t)
    depth = np.where(hit, t, 0.0)                   # => t is exactly the z-depth
    obj = np.where(hit & (gid < len(ids)), gid, -1)
    return depth, obj, nrm, d_w


def make_session(out: str, mask_dropout: int = 0, seed: int = 0, n_frames: int = N_FRAMES):
    rng = np.random.default_rng(seed)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "frames"))
    poses = walkthrough()
    if n_frames != len(poses):
        idx = np.linspace(0, len(poses) - 1, n_frames).round().astype(int)
        poses = [poses[k] for k in idx]
    x0, y0, yaw0 = poses[0]
    T_ra = room_to_arkit(x0, y0, yaw0)
    T_ar = np.linalg.inv(T_ra)

    # like the iOS app: depth intrinsics = RGB intrinsics scaled by the width/height ratios,
    # no half-pixel shift
    sx, sy = DEPTH_W / RGB_W, DEPTH_H / RGB_H
    K_rgb = dict(fx=RGB_FX, fy=RGB_FX, cx=RGB_W / 2, cy=RGB_H / 2, width=RGB_W, height=RGB_H)
    K_d = dict(fx=RGB_FX * sx, fy=RGB_FX * sy, cx=K_rgb["cx"] * sx, cy=K_rgb["cy"] * sy,
               width=DEPTH_W, height=DEPTH_H)
    json.dump({"device": "iphone", "model": "synthetic (tools/fake_session.py)", "app_version": "0",
               "started": "2026-10-24T13:10:00Z", "coord": "arkit_y_up", "depth_units": "meters",
               "has_person_mask": True}, open(os.path.join(out, "session.json"), "w"), indent=2)

    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    statics = static_parts()
    person_track = []
    for i, (x, y, yaw) in enumerate(poses):
        px, py = person_at(round(i * (N_FRAMES - 1) / max(1, n_frames - 1)), (x, y, yaw))
        person_track.append([float(px), float(py)])
        parts = statics + person_parts(px, py)
        scene = o3d.t.geometry.RaycastingScene()
        names = []
        for name, m in parts:
            m = o3d.geometry.TriangleMesh(m).transform(T_ra)                       # room -> ARKit
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
            names.append(name)
        T_cw = T_ra @ gl_cam_pose(x, y, yaw, pitch=PITCH - (math.radians(22) if i % 2 else 0))  # tilt down every other frame                                    # camera -> ARKit
        # --- RGB (flat colour per object, Lambert shading, light in room frame)
        _, obj, nrm, _ = render(scene, names, T_cw, **{k: K_rgb[k] for k in ("fx", "fy", "cx", "cy")},
                                w=RGB_W, h=RGB_H)
        n_room = nrm @ T_ar[:3, :3].T
        shade = 0.45 + 0.55 * np.abs(n_room @ light)
        rgb = np.zeros((RGB_H, RGB_W, 3))
        for k, name in enumerate(names):
            rgb[obj == k] = COLORS[name]
        rgb = np.clip(rgb * shade[..., None] + rng.normal(0, 3, rgb.shape), 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(out, "frames", f"{i:06d}.jpg"), quality=85)
        # --- depth / confidence / person mask at depth resolution
        depth, obj, _, _ = render(scene, names, T_cw, **{k: K_d[k] for k in ("fx", "fy", "cx", "cy")},
                                  w=DEPTH_W, h=DEPTH_H)
        depth = depth + rng.normal(0, 1, depth.shape) * (0.004 * depth ** 2)
        conf = np.full(depth.shape, 2, np.uint8)
        conf[depth > 3.5] = 1
        conf[rng.random(depth.shape) < 0.02] = 0
        depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        depth_mm[conf == 0] = np.where(rng.random(depth_mm[conf == 0].shape) < 0.5, depth_mm[conf == 0], 0)
        is_person = np.isin(obj, [k for k, n in enumerate(names) if n.startswith("person")])
        mask = (is_person * 255).astype(np.uint8)
        if mask_dropout and i % mask_dropout == mask_dropout - 1:
            mask[:] = 0                                                           # segmenter missed
        cv2.imwrite(os.path.join(out, "frames", f"{i:06d}_depth.png"), depth_mm)
        cv2.imwrite(os.path.join(out, "frames", f"{i:06d}_conf.png"), conf)
        cv2.imwrite(os.path.join(out, "frames", f"{i:06d}_mask.png"), mask)
        meta = {"index": i, "timestamp": round(i / 2.5, 3), "pose": T_cw.tolist(),
                "intrinsics_rgb": K_rgb, "intrinsics_depth": K_d,
                "tracking": "limited" if i == 13 else "normal"}
        json.dump(meta, open(os.path.join(out, "frames", f"{i:06d}.json"), "w"))
    gt = {"room_to_arkit": T_ra.tolist(),
          "bag": {"center": [*BAG["c"], BAG["h"] / 2], "size": [*BAG["s"], BAG["h"]]},
          "person_track": person_track, "person_radius": PERSON_R,
          "counter": {"center": [*COUNTER["c"], COUNTER["h"] / 2], "size": [*COUNTER["s"], COUNTER["h"]]},
          "couch": {"center": [*COUCH["c"], 0.4], "size": [*COUCH["s"], COUCH["h"]]},
          "door": DOOR, "walls": {"outer": [0, 0, 8, 6], "inner_x": 4.5},
          "living_room_point": [2.0, 3.0], "kitchen_point": [6.5, 3.0],
          "mask_dropout": mask_dropout}
    json.dump(gt, open(os.path.join(out, "gt.json"), "w"), indent=2)
    print(f"wrote {n_frames} keyframes -> {out}")
    return out


def zip_session(folder: str) -> str:
    base = folder.rstrip("/")
    path = shutil.make_archive(base, "zip", root_dir=os.path.dirname(base) or ".",
                               base_dir=os.path.basename(base))
    print(f"zipped -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--mask-dropout", type=int, default=0,
                    help="drop the person mask on every Nth frame (tests TSDF carving)")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    make_session(a.out, a.mask_dropout, n_frames=a.frames)
    if a.zip:
        zip_session(a.out)


if __name__ == "__main__":
    sys.exit(main())
