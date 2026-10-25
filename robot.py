"""Thin K1 robot interface with a safety layer and swappable backends.

    from robot import make_robot
    r = make_robot("sim")          # MuJoCo K1 (macOS OK)
    r = make_robot("booster_sdk")  # real K1 from the Mac over DDS (community booster-sdk)
    r = make_robot("boosteros")    # real K1, needs Linux + `pip install boosteros` (official)
    r = make_robot("remote", url="http://<linux-box>:8765")  # Mac -> robot_server.py

    r.walk(0.3, 0, 0, seconds=2); r.turn(45); r.look(20, 10); r.wave(); r.say("hi"); r.stop()

SafeRobot enforces, no matter what the model asks for:
  * |v| <= 0.5 m/s (event cap), vx in [-0.3, 0.5], |vy| <= 0.3, |yaw rate| <= 0.8 rad/s
  * every walk/turn has a deadline (<= 3 s); a watchdog zeroes velocity after it
  * motion commands are rate-limited to one per `min_interval` s (default 1.0);
    stop() is never rate-limited
  * head yaw +-45 deg, pitch -17..+45 deg (real K1 limits are a bit wider)
"""
from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import threading
import time

import numpy as np

# ---------------------------------------------------------------- limits
VX_MAX, VX_MIN, VY_MAX, SPEED_MAX = 0.5, -0.3, 0.3, 0.5
VX_DEADBAND = 0.2          # |vx| below this doesn't produce steps in the sim policy
YAW_RATE_MAX = 0.8          # rad/s
TURN_RATE = 0.6             # rad/s used by turn()
MAX_MOVE_S = 3.0
HEAD_YAW_MAX = math.radians(45)
HEAD_PITCH_MIN, HEAD_PITCH_MAX = math.radians(-17), math.radians(45)


def _clip(x, lo, hi):
    return max(lo, min(hi, float(x)))


# ---------------------------------------------------------------- local TTS
def local_tts(text: str, wav_path: str | None = None):
    """On-device speech. macOS: `say`; Linux: espeak-ng/espeak/piper if installed.
    Returns wav path if wav_path was given and synthesis succeeded, else None."""
    if platform.system() == "Darwin" and shutil.which("say"):
        if wav_path:
            subprocess.run(["say", "-o", wav_path, "--data-format=LEI16@16000", text], check=False)
            return wav_path if os.path.exists(wav_path) else None
        subprocess.Popen(["say", text])
        return None
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            if wav_path:
                subprocess.run([exe, "-w", wav_path, text], check=False)
                return wav_path if os.path.exists(wav_path) else None
            subprocess.Popen([exe, text])
            return None
    print(f"[say] (no local TTS found) {text}")
    return None


# ---------------------------------------------------------------- backends
class SimBackend:
    """MuJoCo K1 running Booster's own walking policy (booster_deploy k1_walk)."""

    name = "sim"

    def __init__(self, viewer: bool = False, speak: bool = True):
        from k1_sim import K1Sim

        self.sim = K1Sim(viewer=viewer).start()
        self.speak = speak
        time.sleep(1.0)  # let it settle on its feet

    def set_velocity(self, vx, vy, vyaw):
        self.sim.set_velocity(vx, vy, vyaw)

    def set_head(self, yaw, pitch):
        self.sim.set_head(yaw, pitch)

    def wave(self):
        self.sim.wave(2.5)

    def say(self, text):
        print(f"[K1 says] {text}")
        if self.speak:
            local_tts(text)

    def get_frame(self) -> np.ndarray:
        return self.sim.render("head")

    def get_chase_frame(self) -> np.ndarray:
        return self.sim.render("chase")

    def status(self) -> dict:
        x, y, yaw, z = self.sim.pose()
        return {"x": round(x, 2), "y": round(y, 2), "yaw_deg": round(math.degrees(yaw)), "fallen": self.sim.fallen}

    def close(self):
        self.sim.close()


class BoosterOSBackend:
    """Real K1 via BoosterOS (`pip install boosteros`).

    BoosterOS depends on booster-robotics-sdk-python, which ships only
    manylinux wheels (x86_64/aarch64) and talks DDS/ROS 2 Humble: run this on
    the K1's Jetson or an Ubuntu 22.04 box on the robot network, not on macOS.
    API used (boosteros 1.2.x, see docs.booster.tech BoosterOS Python SDK):
      K1() ; set_mode("walk") ; set_velocity(vx, vy, vyaw)
      set_head_angle(pitch, yaw)   # rad, pitch + = down, yaw + = left, walk mode only
      do_action("hand_wave")       # returns TaskHandle
      get_image() -> Image (BGR internally; .to_pil() gives RGB)
      play_sound(wav_path)
    UNTESTED on hardware from this Mac: verify at the Ultimate Bots table.
    """

    name = "boosteros"

    def __init__(self, speak: bool = True):
        from boosteros.robots.booster import K1  # Linux only

        self.r = K1()
        if self.r.get_mode() != "walk":
            self.r.set_mode("walk")  # robot must already be standing (prepare -> walk per K1 manual)
        self.speak = speak
        self._wave = None

    def set_velocity(self, vx, vy, vyaw):
        self.r.set_velocity(vx, vy, vyaw)

    def set_head(self, yaw, pitch):
        self.r.set_head_angle(pitch, yaw)  # NB argument order is (pitch, yaw)

    def wave(self):
        self._wave = self.r.do_action("hand_wave")
        # hand_wave has duration -1 (runs until cancelled): stop it after ~3 s
        threading.Timer(3.0, lambda: self._wave and self._wave.cancel()).start()

    def say(self, text):
        # Keep speech on-device: synthesize locally and play on the robot speaker.
        # (boosteros.brain.speech is a cloud provider, so it's deliberately not used.)
        print(f"[K1 says] {text}")
        if not self.speak:
            return
        wav = local_tts(text, "/tmp/k1_say.wav")
        if wav:
            self.r.play_sound(wav)

    def get_frame(self) -> np.ndarray:
        return np.asarray(self.r.get_image().to_pil().convert("RGB"))

    def status(self) -> dict:
        try:
            return {"mode": self.r.get_mode()}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def close(self):
        try:
            self.r.set_velocity(0, 0, 0)
        finally:
            close = getattr(self.r, "close", None)
            if close:
                close()


class BoosterSdkBackend:
    """Real K1 straight from the Mac via the community `booster-sdk` (PyPI,
    IntelligentRoboticsLab/booster_sdk, Apache-2.0, Rust + Python, has macOS arm64
    wheels). Speaks the Booster SDK's DDS RPC directly, so the Mac must be on the
    robot's network (Ethernet dongle recommended so DDS discovery works).
    Installed in .venv; constructing works offline, RPCs time out (5 s) with no robot.
      BoosterClient().change_mode(RobotMode.WALKING); move_robot(vx, vy, vyaw)
      rotate_head(pitch, yaw); wave_hand(HandAction.OPEN / CLOSE); play_sound(path ON ROBOT)
    It has no camera-image API (images are a ROS topic), so give frame_url
    (e.g. robot_server.py on the robot -> http://<robot>:8765/frame.jpg) or use
    brain.py --camera webcam.  UNTESTED against a real K1.
    """

    name = "booster_sdk"

    def __init__(self, frame_url: str | None = None, speak: bool = True):
        from booster_sdk.client.booster import BoosterClient, HandAction, RobotMode

        self.HandAction = HandAction
        self.c = BoosterClient()
        try:
            mode = self.c.get_mode()
            print(f"[booster_sdk] current mode: {mode}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"K1 not reachable over DDS ({e}). On the robot network? Ethernet dongle?") from e
        self.c.change_mode(RobotMode.WALKING)  # robot must already be standing/prepared
        self.frame_url = frame_url
        self.speak = speak

    def set_velocity(self, vx, vy, vyaw):
        self.c.move_robot(vx, vy, vyaw)

    def set_head(self, yaw, pitch):
        self.c.rotate_head(pitch, yaw)  # (pitch, yaw), rad, pitch + = down

    def wave(self):
        self.c.wave_hand(self.HandAction.OPEN)
        threading.Timer(3.0, lambda: self.c.wave_hand(self.HandAction.CLOSE)).start()

    def say(self, text):
        # Laptop speaker, on-device TTS. (play_sound() takes a path on the ROBOT's disk.)
        print(f"[K1 says] {text}")
        if self.speak:
            local_tts(text)

    def get_frame(self) -> np.ndarray:
        if not self.frame_url:
            raise RuntimeError("booster_sdk has no image API: pass frame_url or use --camera webcam")
        import io

        import requests
        from PIL import Image

        r = requests.get(self.frame_url, timeout=3)
        return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))

    def status(self) -> dict:
        try:
            return {"mode": str(self.c.get_mode())}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def close(self):
        try:
            self.c.move_robot(0, 0, 0)
        except Exception:  # noqa: BLE001
            pass


class RemoteBackend:
    """Mac-side client for robot_server.py (which runs BoosterOSBackend on Linux)."""

    name = "remote"

    def __init__(self, url: str, speak_locally: bool = False):
        import requests  # noqa: F401

        self.url = url.rstrip("/")
        self.speak_locally = speak_locally

    def _post(self, path, **kw):
        import requests

        r = requests.post(f"{self.url}/{path}", json=kw, timeout=3)
        r.raise_for_status()
        return r.json()

    def set_velocity(self, vx, vy, vyaw):
        self._post("velocity", vx=vx, vy=vy, vyaw=vyaw)

    def set_head(self, yaw, pitch):
        self._post("head", yaw=yaw, pitch=pitch)

    def wave(self):
        self._post("wave")

    def say(self, text):
        if self.speak_locally:
            local_tts(text)
        self._post("say", text=text)

    def get_frame(self) -> np.ndarray:
        import io

        import requests
        from PIL import Image

        r = requests.get(f"{self.url}/frame.jpg", timeout=3)
        r.raise_for_status()
        return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))

    def status(self) -> dict:
        import requests

        return requests.get(f"{self.url}/status", timeout=3).json()

    def close(self):
        try:
            self._post("velocity", vx=0, vy=0, vyaw=0)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- safety layer
class SafeRobot:
    def __init__(self, backend, min_interval: float = 1.0):
        self.b = backend
        self.min_interval = min_interval
        self._last_cmd = 0.0
        self._lock = threading.Lock()
        self._deadline_timer: threading.Timer | None = None
        self.log: list[str] = []

    # -- helpers
    def _rate_limit(self):
        wait = self._last_cmd + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_cmd = time.monotonic()

    def _arm_deadline(self, seconds):
        if self._deadline_timer:
            self._deadline_timer.cancel()
        self._deadline_timer = threading.Timer(seconds, lambda: self.b.set_velocity(0, 0, 0))
        self._deadline_timer.daemon = True
        self._deadline_timer.start()

    def _note(self, s):
        self.log.append(s)
        print(f"[robot] {s}")
        return s

    # -- public API
    def walk(self, vx: float, vy: float = 0.0, yaw: float = 0.0, seconds: float = 1.5) -> str:
        """vx, vy in m/s, yaw in rad/s; auto-stops after `seconds` (<= 3)."""
        with self._lock:
            vx = _clip(vx, VX_MIN, VX_MAX)
            vy = _clip(vy, -VY_MAX, VY_MAX)
            if 0 < abs(vx) < VX_DEADBAND:  # walking policy ignores tiny commands
                vx = math.copysign(VX_DEADBAND, vx)
            sp = math.hypot(vx, vy)
            if sp > SPEED_MAX:
                vx, vy = vx * SPEED_MAX / sp, vy * SPEED_MAX / sp
            yaw = _clip(yaw, -YAW_RATE_MAX, YAW_RATE_MAX)
            seconds = _clip(seconds, 0.2, MAX_MOVE_S)
            self._rate_limit()
            self.b.set_velocity(vx, vy, yaw)
            self._arm_deadline(seconds)
            return self._note(f"walk vx={vx:.2f} vy={vy:.2f} yaw={yaw:.2f} for {seconds:.1f}s")

    def turn(self, degrees: float) -> str:
        """Turn in place; + = left. Open-loop (rate x time), so expect +-20% error."""
        with self._lock:
            degrees = _clip(degrees, -90, 90)
            rate = math.copysign(TURN_RATE, degrees)
            seconds = _clip(abs(math.radians(degrees)) / TURN_RATE, 0.2, MAX_MOVE_S)
            self._rate_limit()
            self.b.set_velocity(0.0, 0.0, rate)
            self._arm_deadline(seconds)
            return self._note(f"turn {degrees:.0f} deg ({rate:+.2f} rad/s for {seconds:.1f}s)")

    def look(self, yaw_deg: float, pitch_deg: float) -> str:
        """Head angles in degrees; yaw + = left, pitch + = down."""
        with self._lock:
            y = _clip(math.radians(yaw_deg), -HEAD_YAW_MAX, HEAD_YAW_MAX)
            p = _clip(math.radians(pitch_deg), HEAD_PITCH_MIN, HEAD_PITCH_MAX)
            self._rate_limit()
            self.b.set_head(y, p)
            return self._note(f"look yaw={math.degrees(y):.0f} pitch={math.degrees(p):.0f}")

    def wave(self) -> str:
        with self._lock:
            self._rate_limit()
            self.b.set_velocity(0, 0, 0)  # don't wave while walking
            self.b.wave()
            return self._note("wave")

    def say(self, text: str) -> str:
        text = str(text)[:200]
        self.b.say(text)  # speech isn't motion: not rate-limited
        return self._note(f"say {text!r}")

    def stop(self) -> str:
        if self._deadline_timer:
            self._deadline_timer.cancel()
        self.b.set_velocity(0, 0, 0)  # never rate-limited, never blocked by the lock
        return self._note("stop")

    def get_frame(self) -> np.ndarray:
        return self.b.get_frame()

    def status(self) -> dict:
        return self.b.status()

    def close(self):
        self.stop()
        self.b.close()

    # dispatch a model tool call {"name":..., "arguments": {...}}
    def call(self, name: str, args: dict) -> str:
        args = dict(args or {})
        try:
            if name == "walk":
                return self.walk(args.get("vx", 0), args.get("vy", 0), args.get("yaw", 0),
                                 args.get("seconds", 1.5))
            if name == "turn":
                return self.turn(args.get("degrees", 0))
            if name == "look":
                return self.look(args.get("yaw_deg", 0), args.get("pitch_deg", 0))
            if name == "wave":
                return self.wave()
            if name == "say":
                return self.say(args.get("text", ""))
            if name == "stop":
                return self.stop()
            return self._note(f"ignored unknown tool {name!r}")
        except (TypeError, ValueError) as e:
            self.stop()
            return self._note(f"bad args for {name}: {e}; stopped")


def make_robot(kind: str = "sim", **kw) -> SafeRobot:
    min_interval = kw.pop("min_interval", 1.0)
    if kind == "sim":
        b = SimBackend(**kw)
    elif kind == "boosteros":
        b = BoosterOSBackend(**kw)
    elif kind == "remote":
        b = RemoteBackend(**kw)
    elif kind == "booster_sdk":
        b = BoosterSdkBackend(**kw)
    else:
        raise ValueError(kind)
    return SafeRobot(b, min_interval=min_interval)


if __name__ == "__main__":
    r = make_robot("sim", speak=False)
    print(r.status())
    r.walk(2.0, 0, 0, seconds=10)   # clamped to 0.5 m/s, 3 s
    time.sleep(3.5)
    print("after walk:", r.status())
    r.turn(90)
    time.sleep(3.0)
    print("after turn 90:", r.status())
    r.look(30, 15); r.wave(); time.sleep(3)
    r.say("hello from the sim")
    r.stop()
    print("final:", r.status())
    r.close()
