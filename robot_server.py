"""Tiny HTTP bridge: run next to the real K1 (on its Jetson or an Ubuntu 22.04
laptop on the robot network, where BoosterOS works), drive it from the Mac.

    # on the Linux box (once):  python3 -m pip install boosteros requests pillow numpy opencv-python-headless
    python3 robot_server.py --backend boosteros --port 8765
    # on the Mac:
    .venv/bin/python brain.py --robot remote --url http://<linux-box-ip>:8765
    .venv/bin/python navigator.py maps/agihouse --robot real --url http://<linux-box-ip>:8765 --goal restroom

    # test the whole remote path on the Mac:
    .venv/bin/python robot_server.py --backend sim         # MuJoCo K1 (gestures/camera, no /odom or /scan)
    .venv/bin/python robot_server.py --dry-run             # fake K1 in the AGI House map: /odom, /scan, /drive

Endpoints
  GET  /status /frame.jpg
  GET  /odom   -> {x, y, th, t, fallen}           leg odometry (BoosterOS get_odom = SDK rt/odom)
  GET  /scan   -> {ranges, angles, t, odom, floor} horizontal depth slice, robot frame, left +, null = no return
  POST /velocity {vx, vy, vyaw}   3 s watchdog      POST /drive {vx, vyaw}   0.5 s deadman, no rate limit
  POST /head /wave /say /stop /reset_odom

The scan: head depth image (BoosterOS get_image(img_type="depth") = ROS /boostercamera/head/depth, or
--depth-source ros/sgbm) -> camera-frame points -> floor plane fitted per frame (gives camera height and
pitch while the body sways; falls back to --cam-height/--cam-pitch) -> keep points 0.15-1.1 m above the
floor -> min horizontal range per bearing bin, <= 5 m.

Defence in depth: this side clamps again and zeroes velocity if no velocity
command arrives for 3 s (/velocity) or 0.5 s (/drive), so a crashed/disconnected
brain can't leave it walking. Hardware deps import lazily. No auth: keep it on the robot LAN.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import robot as R

WATCHDOG_S = 3.0
DEADMAN_S = 0.5
BAND = (0.15, 1.1)          # m above the floor kept in the slice
SCAN_MAX = 5.0
BEAMS = 90


class State:
    backend = None
    sensors = None
    last_vel = 0.0
    last_drive = 0.0
    quiet = True
    lock = threading.Lock()


def clamp_vel(vx, vy, vyaw):
    vx = R._clip(vx, R.VX_MIN, R.VX_MAX)
    vy = R._clip(vy, -R.VY_MAX, R.VY_MAX)
    sp = math.hypot(vx, vy)
    if sp > R.SPEED_MAX:
        vx, vy = vx * R.SPEED_MAX / sp, vy * R.SPEED_MAX / sp
    return vx, vy, R._clip(vyaw, -R.YAW_RATE_MAX, R.YAW_RATE_MAX)


def watchdog():
    while True:
        time.sleep(0.05)
        with State.lock:
            now = time.monotonic()
            if State.last_vel and now - State.last_vel > WATCHDOG_S:
                State.backend.set_velocity(0, 0, 0)
                State.last_vel = 0.0
                print("[watchdog] no velocity command for 3 s -> stopped", flush=True)
            if State.last_drive and now - State.last_drive > DEADMAN_S:
                State.backend.set_velocity(0, 0, 0)
                State.last_drive = 0.0
                print("[deadman] no /drive for 0.5 s -> stopped", flush=True)


# ---------------------------------------------------------------- depth image -> horizontal slice
def fit_floor(z, x, y, cam_h, pitch, tols=(0.15, 0.06, 0.04, 0.04)):
    """Least-squares floor plane from points near the expected floor. Returns (cam_h, pitch) or None.
    Camera frame: x right, y down, z forward; pitch + = looking down."""
    for tol in tols:  # shrinking band keeps wall bases out of the fit
        down = y * math.cos(pitch) + z * math.sin(pitch)
        sel = (np.abs(cam_h - down) < tol) & (z < 4.0)
        if sel.sum() < 300:
            return None
        P = np.column_stack([x[sel], y[sel], z[sel]])
        c = P.mean(0)
        n = np.linalg.svd(P - c, full_matrices=False)[2][2]
        n = n if n[1] > 0 else -n                       # normal pointing down (toward the floor)
        cam_h, pitch = float(n @ c), math.atan2(n[2], n[1])
    return (cam_h, pitch) if 0.4 < cam_h < 1.3 and abs(pitch) < math.radians(50) else None


def depth_slice(depth, K, cam_h, pitch, beams=BEAMS, band=BAND, max_range=SCAN_MAX, fit=True):
    """depth: HxW metres (0/nan invalid), K = (fx, fy, cx, cy). Returns ranges, angles, floor-info."""
    fx, fy, cx, cy = K
    H, W = depth.shape
    try:
        import cv2

        depth = cv2.medianBlur(np.nan_to_num(depth, nan=0.0).astype(np.float32), 5)
    except ImportError:
        pass
    v, u = np.mgrid[0:H:2, 0:W:2]
    z = depth[::2, ::2].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.1) & (z < 10.0)
    z, x, y = z[ok], ((u[ok] - cx) / fx) * z[ok], ((v[ok] - cy) / fy) * z[ok]
    got = fit_floor(z, x, y, cam_h, pitch) if fit else None
    if got:
        cam_h, pitch = got
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd, hgt = z * cp - y * sp, cam_h - (y * cp + z * sp)
    rng, bear = np.hypot(fwd, x), np.arctan2(-x, fwd)
    keep = (hgt >= band[0]) & (hgt <= band[1]) & (fwd > 0.05)
    half = math.atan2(max(cx, W - cx), fx)
    edges = np.linspace(-half, half, beams + 1)
    b = np.digitize(bear[keep], edges) - 1
    inb = (b >= 0) & (b < beams)
    out = np.full(beams, np.inf)
    np.minimum.at(out, b[inb], rng[keep][inb])
    cnt = np.bincount(b[inb], minlength=beams)
    out[(cnt < 3) | (out > max_range)] = np.nan
    return out, (edges[:-1] + edges[1:]) / 2, {"cam_h": round(cam_h, 3), "pitch_deg": round(math.degrees(pitch), 1),
                                                 "fit": bool(got)}


def depth_to_metres(arr, encoding):
    arr = np.asarray(arr).squeeze()
    if arr.dtype == np.uint16 or "16u" in encoding.lower() or "mono16" in encoding.lower():
        return arr.astype(np.float32) / 1000.0
    return arr.astype(np.float32)


def scale_K(K, info_wh, img_wh):
    sx, sy = img_wh[0] / info_wh[0], img_wh[1] / info_wh[1]
    return (K[0] * sx, K[1] * sy, K[2] * sx, K[3] * sy)


# ---------------------------------------------------------------- sensors on the real K1
class RosImages:
    """Direct rclpy subscriptions (for --depth-source ros / sgbm, or if BoosterOS image calls misbehave)."""

    def __init__(self, topics):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        try:
            rclpy.init()
        except RuntimeError:
            pass  # BoosterOS already initialised rclpy in this process
        self.node = rclpy.create_node("k1_scan_bridge")
        self.msgs = {}
        for key, (topic, typ) in topics.items():
            self.node.create_subscription(CameraInfo if typ == "info" else Image, topic,
                                          lambda m, k=key: self.msgs.__setitem__(k, m), qos_profile_sensor_data)
        from rclpy.executors import SingleThreadedExecutor

        ex = SingleThreadedExecutor()
        ex.add_node(self.node)
        threading.Thread(target=ex.spin, daemon=True).start()

    def wait(self, key, timeout=2.0):
        t_end = time.monotonic() + timeout
        while key not in self.msgs:
            if time.monotonic() > t_end:
                raise RuntimeError(f"no message on {key} topic yet")
            time.sleep(0.02)
        return self.msgs[key]

    @staticmethod
    def to_numpy(msg):
        enc = msg.encoding.lower()
        dt = {"16uc1": np.uint16, "mono16": np.uint16, "32fc1": np.float32}.get(enc, np.uint8)
        ch = 3 if enc in ("rgb8", "bgr8") else 4 if enc in ("rgba8", "bgra8") else 1
        a = np.frombuffer(bytes(msg.data), dtype=dt)
        if msg.is_bigendian:
            a = a.byteswap()
        row = msg.step // np.dtype(dt).itemsize
        a = a.reshape(msg.height, row)[:, :msg.width * ch].reshape(msg.height, msg.width, ch).squeeze()
        return a, enc


class K1Sensors:
    """Odometry / fall state from BoosterOS, depth from BoosterOS or ROS topics."""

    def __init__(self, k1, a):
        self.k1, self.a, self._K, self.ros = k1, a, None, None
        if a.depth_source == "ros":
            self.ros = RosImages({"depth": (a.depth_topic, "img"), "info": (a.info_topic, "info")})
        elif a.depth_source == "sgbm":
            self.ros = RosImages({"left": (a.left_topic, "img"), "right": (a.right_topic, "img"),
                                  "info": (a.info_topic, "info")})

    def odom(self):
        p = self.k1.get_odom().pose_2d
        return float(p[0]), float(p[1]), float(p[2])

    def reset_odom(self):
        self.k1.reset_odom()

    def fallen(self):
        try:
            return self.k1.get_fall_down_state().state in ("falling", "fallen")
        except Exception:  # noqa: BLE001
            return False

    def _intrinsics(self):
        if self._K is None:
            if self.ros:
                ci = self.ros.wait("info")
                k, wh = np.asarray(ci.k, float).reshape(3, 3), (ci.width, ci.height)
            else:
                ci = self.k1.get_camera_info()
                k, wh = np.asarray(ci.k, float).reshape(3, 3), (ci.width, ci.height)
            self._K = ((k[0, 0], k[1, 1], k[0, 2], k[1, 2]), wh)
            print(f"[scan] intrinsics fx={k[0, 0]:.1f} fy={k[1, 1]:.1f} cx={k[0, 2]:.1f} cy={k[1, 2]:.1f} at {wh}")
        return self._K

    def depth(self):
        K, wh = self._intrinsics()
        if self.a.depth_source == "boosteros":
            img = self.k1.get_image(img_type="depth")
            d = depth_to_metres(img.to_numpy(), getattr(img, "encoding", ""))
        elif self.a.depth_source == "ros":
            d = depth_to_metres(*self.ros.to_numpy(self.ros.wait("depth")))
        else:
            d = self.sgbm(K, wh)
            return d, scale_K(K, wh, d.shape[::-1])
        return d, scale_K(K, wh, d.shape[::-1])

    def sgbm(self, K, wh, width=320):
        """Disparity on (already rectified) left/right images, downscaled for the Orin CPU."""
        import cv2

        imgs = []
        for k in ("left", "right"):
            a, enc = self.ros.to_numpy(self.ros.wait(k))
            g = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY if enc.startswith("rgb") else
                                                   cv2.COLOR_BGR2GRAY)
            imgs.append(cv2.resize(g, (width, int(g.shape[0] * width / g.shape[1]))))
        if not hasattr(self, "_sgbm"):
            self._sgbm = cv2.StereoSGBM_create(0, 64, 5, P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                               speckleWindowSize=100, speckleRange=2,
                                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = self._sgbm.compute(imgs[0], imgs[1]).astype(np.float32) / 16.0
        fx = K[0] * width / wh[0]
        with np.errstate(divide="ignore"):
            return np.where(disp > 0.5, fx * self.a.baseline / disp, np.nan).astype(np.float32)

    def scan(self):
        o, t = self.odom(), time.time()
        d, K = self.depth()
        r, ang, floor = depth_slice(d, K, self.a.cam_height, math.radians(self.a.cam_pitch))
        return r, ang, t, o, floor


# ---------------------------------------------------------------- dry run: fake K1 in the scanned map
class DryBackend:
    """Integrates commanded velocity (with lag, yaw under-tracking and odometry drift) inside the map;
    renders a synthetic head depth image by raycasting the floor plan, so /scan exercises depth_slice."""

    name = "dry"

    def __init__(self, map_dir, start, yaw_deg, cam_h=0.88, pitch_deg=8.0, seed=0):
        import navigator as N
        import scan_map

        self.N, self.m = N, scan_map.Map(map_dir)
        self.field = N.Field(self.m)
        x, y = self.m.place(start)
        self.pose = (x, y, math.radians(yaw_deg))   # true map pose (never served)
        self.odom = (0.0, 0.0, 0.0)
        self.cmd, self.vel = (0.0, 0.0, 0.0), np.zeros(3)
        self.rng = np.random.default_rng(seed)
        self.k_scale, self.k_rot, self.drift = 1.03, 0.97, math.radians(1.0)
        self.cam_h, self.pitch = cam_h, math.radians(pitch_deg)
        self.W, self.H = 320, 240
        f = (self.W / 2) / math.tan(math.radians(105 / 2))
        self.K = (f, f, self.W / 2, self.H / 2)
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self, dt=0.01):
        N = self.N
        while True:
            time.sleep(dt)
            with self.lock:
                self.vel += (np.array(self.cmd) - self.vel) * min(1.0, dt / 0.25)   # gait lag
                vx, vy, w = self.vel
                d = (vx * dt, vy * dt, 0.92 * w * dt)                              # yaw under-tracks
                self.pose = N.compose(self.pose, d)
                dist = math.hypot(d[0], d[1])
                self.odom = N.compose(self.odom, (d[0] * self.k_scale, d[1] * self.k_scale,
                                                  d[2] * self.k_rot + self.drift * dist))

    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.cmd = (vx, vy, vyaw)

    def set_head(self, yaw, pitch):
        pass

    def wave(self):
        print("[dry] wave")

    def say(self, text):
        print(f"[dry] K1 says {text}")

    def get_frame(self):
        d, _ = self.depth()
        return (255 * (1 - np.clip(np.nan_to_num(d, nan=0) / 6, 0, 1))).astype(np.uint8)[..., None].repeat(3, 2)

    def status(self):
        with self.lock:
            return {"backend": "dry", "cmd": self.cmd, "true_pose": [round(v, 3) for v in self.pose]}  # for scoring

    def close(self):
        self.set_velocity(0, 0, 0)

    # sensors
    def reset_odom(self):
        with self.lock:
            self.odom = (0.0, 0.0, 0.0)

    def fallen(self):
        return False

    def depth(self):
        fx, fy, cx, cy = self.K
        with self.lock:
            pose = self.pose
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        xn, yn = (u - cx) / fx, (v - cy) / fy
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fwd, down = cp - yn * sp, yn * cp + sp
        bear = np.arctan2(-xn, fwd)
        n = 241
        fov = 2 * math.atan2(cx, fx) + 0.02
        r, ang = self.field.raycast(pose, None, fov=fov, n=n)
        r = np.nan_to_num(r, nan=50.0)
        dh = np.interp(bear, ang, r)
        t_wall = dh / np.hypot(fwd, xn)
        with np.errstate(divide="ignore"):
            t_floor = np.where(down > 1e-3, self.cam_h / down, np.inf)
        z = np.minimum(t_wall, t_floor) * (1 + 0.02 * self.rng.standard_normal((self.H, self.W)))
        z[(z > 8) | (self.rng.random(z.shape) < 0.1)] = np.nan
        return z.astype(np.float32), self.K

    def scan(self):
        with self.lock:
            o = self.odom
        t = time.time()
        d, K = self.depth()
        r, ang, floor = depth_slice(d, K, 0.8, 0.0)   # deliberately wrong prior: the floor fit must fix it
        return r, ang, t, o, floor


class DrySensors:
    def __init__(self, b):
        self.b = b

    def odom(self):
        with self.b.lock:
            return self.b.odom

    def reset_odom(self):
        self.b.reset_odom()

    def fallen(self):
        return False

    def scan(self):
        return self.b.scan()


# ---------------------------------------------------------------- HTTP
def _nulls(a, nd=3):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        b, s = State.backend, State.sensors
        try:
            if self.path.startswith("/frame.jpg"):
                from PIL import Image

                buf = io.BytesIO()
                Image.fromarray(b.get_frame()).save(buf, format="JPEG", quality=85)
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/status"):
                self._json(b.status())
            elif self.path.startswith("/odom"):
                if not s:
                    return self._json({"error": "no odometry on this backend"}, 501)
                x, y, th = s.odom()
                self._json({"x": round(x, 4), "y": round(y, 4), "th": round(th, 5), "t": time.time(),
                            "fallen": bool(s.fallen())})
            elif self.path.startswith("/scan"):
                if not s:
                    return self._json({"error": "no depth on this backend"}, 501)
                r, ang, t, o, floor = s.scan()
                self._json({"ranges": _nulls(r), "angles": _nulls(ang, 4), "t": t,
                            "odom": [round(v, 5) for v in o], "floor": floor,
                            "valid": int(np.isfinite(r).sum())})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        args = json.loads(self.rfile.read(n) or b"{}")
        b = State.backend
        try:
            with State.lock:
                if self.path == "/drive":
                    vx, _, vyaw = clamp_vel(args.get("vx", 0), 0, args.get("vyaw", 0))
                    b.set_velocity(vx, 0, vyaw)
                    State.last_drive = time.monotonic() if (vx or vyaw) else 0.0
                    State.last_vel = 0.0
                    return self._json({"ok": True, "vx": vx, "vyaw": vyaw})
                if self.path == "/velocity":
                    vx, vy, vyaw = clamp_vel(args.get("vx", 0), args.get("vy", 0), args.get("vyaw", 0))
                    b.set_velocity(vx, vy, vyaw)
                    State.last_vel = time.monotonic() if (vx or vy or vyaw) else 0.0
                    State.last_drive = 0.0
                    return self._json({"ok": True, "vx": vx, "vy": vy, "vyaw": vyaw})
                if self.path == "/head":
                    y = R._clip(args.get("yaw", 0), -R.HEAD_YAW_MAX, R.HEAD_YAW_MAX)
                    p = R._clip(args.get("pitch", 0), R.HEAD_PITCH_MIN, R.HEAD_PITCH_MAX)
                    b.set_head(y, p)
                    return self._json({"ok": True})
                if self.path == "/wave":
                    b.set_velocity(0, 0, 0)
                    b.wave()
                    return self._json({"ok": True})
                if self.path == "/say":
                    b.say(str(args.get("text", ""))[:200])
                    return self._json({"ok": True})
                if self.path == "/stop":
                    b.set_velocity(0, 0, 0)
                    State.last_vel = State.last_drive = 0.0
                    return self._json({"ok": True})
                if self.path == "/reset_odom":
                    if not State.sensors:
                        return self._json({"error": "no odometry on this backend"}, 501)
                    State.sensors.reset_odom()
                    return self._json({"ok": True})
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            b.set_velocity(0, 0, 0)
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *a):
        if State.quiet and ("/odom" in self.requestline or "/drive" in self.requestline):
            return
        print("[http]", fmt % a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="boosteros", choices=["boosteros", "sim"])
    ap.add_argument("--dry-run", action="store_true", help="fake K1 in --map: test every endpoint on the Mac")
    ap.add_argument("--map", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "agihouse"))
    ap.add_argument("--dry-start", default="front_entrance")
    ap.add_argument("--dry-yaw", type=float, default=90.0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--depth-source", default="boosteros", choices=["boosteros", "ros", "sgbm"])
    ap.add_argument("--depth-topic", default="/boostercamera/head/depth")
    ap.add_argument("--info-topic", default="/boostercamera/head/rgb/camera_info")
    ap.add_argument("--left-topic", default="/boostercamera/head/left/image_rect")
    ap.add_argument("--right-topic", default="/boostercamera/head/right/image_rect")
    ap.add_argument("--baseline", type=float, default=0.06, help="m, stereo baseline (sgbm only)")
    ap.add_argument("--cam-height", type=float, default=0.88, help="m, prior if the floor fit fails")
    ap.add_argument("--cam-pitch", type=float, default=0.0, help="deg down, prior if the floor fit fails")
    ap.add_argument("--verbose", action="store_true", help="also log /odom and /drive requests")
    a = ap.parse_args()
    State.quiet = not a.verbose
    if a.dry_run:
        State.backend = DryBackend(a.map, a.dry_start, a.dry_yaw)
        State.sensors = DrySensors(State.backend)
        kind = f"dry-run in {a.map} at {a.dry_start}"
    elif a.backend == "boosteros":
        State.backend = R.BoosterOSBackend()
        State.sensors = K1Sensors(State.backend.r, a)
        kind = f"boosteros, depth from {a.depth_source}"
    else:
        State.backend = R.SimBackend(speak=False)
        kind = "sim"
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"robot_server ({kind}) on http://{a.host}:{a.port}")
    try:
        ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    finally:
        State.backend.set_velocity(0, 0, 0)
        State.backend.close()


if __name__ == "__main__":
    main()
