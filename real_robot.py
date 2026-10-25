"""The real K1 behind robot_server.py, with the same interface as navigator.SimRobot.

    robot = RealRobot("http://<jetson>:8765")
    robot.drive(0.3, 0.0); robot.step(50); r, a = robot.scan(); robot.close()

t is wall-clock seconds since construction; step(n) sleeps n * dt while a background thread polls
/odom at ~20 Hz; drive() is re-sent at 10 Hz while nonzero (the server stops the robot if no /drive
arrives for 0.5 s, so a dead laptop or network stops the K1). scan() returns the slice re-expressed
in the robot's *current* frame using the odometry stamped at capture time (scan latency while
turning would otherwise smear the global match).
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from navigator import compose, inverse

SEND_HZ = 10
ODOM_HZ = 20
ODOM_STALE_S = 2.0


class RealRobot:
    def __init__(self, url, dt=0.02, reset_odom=True, log=print):
        import requests

        self.url, self.dt, self.log, self.frames = url.rstrip("/"), dt, log, []
        self._rq = requests
        self._t0 = time.monotonic()
        self._cmd, self._sent_zero = (0.0, 0.0), True
        self._wake, self._quit = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._main = requests.Session()
        try:
            if reset_odom:
                r = self._main.post(f"{self.url}/reset_odom", json={}, timeout=3)
                if r.status_code != 200:
                    log(f"[real] reset_odom failed ({r.status_code} {r.text[:80]}); using odometry as is")
            o = self._main.get(f"{self.url}/odom", timeout=3).json()
        except requests.RequestException as e:
            raise RuntimeError(f"robot_server not reachable at {self.url}: {e}") from e
        if "x" not in o:
            raise RuntimeError(f"/odom returned {o}")
        self.odom, self._fallen, self._odom_at = (o["x"], o["y"], o["th"]), bool(o.get("fallen")), time.monotonic()
        threading.Thread(target=self._poll, daemon=True).start()
        threading.Thread(target=self._sender, daemon=True).start()

    @property
    def t(self):
        return time.monotonic() - self._t0

    def _poll(self):
        s = self._rq.Session()
        while not self._quit.is_set():
            try:
                o = s.get(f"{self.url}/odom", timeout=0.5).json()
                self.odom, self._fallen, self._odom_at = (o["x"], o["y"], o["th"]), bool(o.get("fallen")), \
                    time.monotonic()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1 / ODOM_HZ)

    def _sender(self):
        s = self._rq.Session()
        while not self._quit.is_set():
            self._wake.wait(1 / SEND_HZ)
            self._wake.clear()
            with self._lock:
                cmd, zero = self._cmd, self._cmd == (0.0, 0.0)
                if zero and self._sent_zero:
                    continue
            try:
                s.post(f"{self.url}/drive", json={"vx": cmd[0], "vyaw": cmd[1]}, timeout=0.3)
                if zero:
                    with self._lock:
                        self._sent_zero = self._cmd == (0.0, 0.0)
            except Exception as e:  # noqa: BLE001
                self.log(f"[real] /drive failed: {e}")

    def step(self, n=1):
        time.sleep(self.dt * n)
        if time.monotonic() - self._odom_at > ODOM_STALE_S:
            self.close()
            raise RuntimeError(f"no odometry from {self.url} for {ODOM_STALE_S} s: stopped")

    def drive(self, vx, vyaw):
        cmd = (float(vx), float(vyaw))
        with self._lock:
            changed = cmd != self._cmd
            self._cmd = cmd
            if cmd == (0.0, 0.0):
                self._sent_zero = False
        if changed:
            self._wake.set()

    def scan(self):
        j = self._main.get(f"{self.url}/scan", timeout=5).json()
        if "ranges" not in j:
            raise RuntimeError(f"/scan returned {j}")
        r = np.array([np.nan if v is None else v for v in j["ranges"]], float)
        a = np.array(j["angles"], float)
        rel = compose(inverse(self.odom), tuple(j["odom"]))   # capture pose in the current robot frame
        ok = np.isfinite(r)
        px = rel[0] + r[ok] * np.cos(a[ok] + rel[2])
        py = rel[1] + r[ok] * np.sin(a[ok] + rel[2])
        r2, a2 = r.copy(), a + rel[2]
        r2[ok], a2[ok] = np.hypot(px, py), np.arctan2(py, px)
        return r2, a2

    @property
    def fallen(self):
        return self._fallen

    def true_pose(self):
        raise NotImplementedError("no ground truth on the real robot")

    # ---- body actions used by escort.py (conversation) ----
    gif = False

    def head_image(self):
        import io

        from PIL import Image

        r = self._main.get(f"{self.url}/frame.jpg", timeout=3)
        return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))

    def wave(self):
        self._main.post(f"{self.url}/wave", json={}, timeout=3)

    def set_head(self, yaw, pitch):
        self._main.post(f"{self.url}/head", json={"yaw": yaw, "pitch": pitch}, timeout=3)

    def say(self, text):
        self._main.post(f"{self.url}/say", json={"text": text}, timeout=3)

    def close(self):
        self._quit.set()
        for _ in range(3):
            try:
                self._main.post(f"{self.url}/stop", json={}, timeout=1)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        self.log("[real] could not send /stop: the server deadman stops the robot within 0.5 s")
