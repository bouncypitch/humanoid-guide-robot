"""MuJoCo twin of the Booster K1, driven by Booster's own walking policy.

Wraps booster_deploy's MujocoController (task "k1_walk") but runs it headless
in a background thread so other code can send velocity / head / arm commands
and grab camera frames. Works on macOS without mjpython; pass viewer=True and
launch with `mjpython` if you want the interactive MuJoCo window.

    sim = K1Sim(); sim.start()
    sim.set_velocity(0.3, 0, 0); time.sleep(3); sim.set_velocity(0, 0, 0)
    rgb = sim.render("head")       # HxWx3 uint8, robot's eye view
    rgb = sim.render("chase")      # third-person view
"""
from __future__ import annotations

import os
import pkgutil
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(HERE, "booster_deploy")
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

import mujoco  # noqa: E402

# Real-robot joint order used by booster_deploy (see booster_deploy/robots/k1.py)
HEAD_YAW, HEAD_PITCH = 0, 1
R_SH_PITCH, R_SH_ROLL, R_EL_PITCH, R_EL_YAW = 6, 7, 8, 9


def _load_tasks():
    import tasks  # booster_deploy/tasks

    for m in pkgutil.walk_packages(tasks.__path__, prefix="tasks."):
        __import__(m.name)


def _build_scene_model(xml_path: str, walls=None, with_props: bool = True) -> mujoco.MjModel:
    """K1 MJCF + a head camera, a chase camera and a few props to look at.

    Props are static geoms (no joints) so qpos layout is unchanged.
    walls: optional list of (cx, cy, half_x, half_y, height) boxes in the sim
    frame (e.g. from scan_map.py); visual only, collisions are checked by the
    caller against the occupancy grid.
    """
    spec = mujoco.MjSpec.from_file(xml_path)
    head = spec.body("aahead_pitch_link")
    # Camera looks along the head's +x; MuJoCo cameras look down their -z.
    head.add_camera(name="head_cam", pos=[0.06, 0.0, 0.09],
                    xyaxes=[0, -1, 0, 0, 0, 1], fovy=80)
    trunk = spec.body("trunk")
    trunk.add_camera(name="chase_cam", pos=[-1.6, -0.9, 0.5],
                     xyaxes=[0.5, -0.866, 0, 0.25, 0.144, 0.958], fovy=60,
                     mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM)
    wb = spec.worldbody
    props = [
        ("red_box", mujoco.mjtGeom.mjGEOM_BOX, [2.0, 0.6, 0.25], [0.25, 0.25, 0.25], [0.9, 0.1, 0.1, 1]),
        ("blue_ball", mujoco.mjtGeom.mjGEOM_SPHERE, [2.5, -1.0, 0.3], [0.3, 0, 0], [0.1, 0.3, 0.95, 1]),
        ("green_pillar", mujoco.mjtGeom.mjGEOM_CYLINDER, [-0.5, 2.5, 0.6], [0.2, 0.6, 0], [0.1, 0.8, 0.2, 1]),
        ("yellow_box", mujoco.mjtGeom.mjGEOM_BOX, [0.5, -2.8, 0.4], [0.4, 0.4, 0.4], [0.95, 0.85, 0.1, 1]),
    ]
    for name, gtype, pos, size, rgba in (props if with_props else []):
        wb.add_geom(name=name, type=gtype, pos=pos, size=size, rgba=rgba,
                    contype=0, conaffinity=0)  # visual only: can't trip the robot
    for i, (cx, cy, hx, hy, h, *rest) in enumerate(walls or []):
        yaw = rest[0] if rest else 0.0
        wb.add_geom(name=f"wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[cx, cy, h / 2], size=[hx, hy, h / 2],
                    quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                    rgba=[0.75, 0.72, 0.68, 1], contype=0, conaffinity=0)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    return spec.compile()


class K1Sim:
    def __init__(self, task: str = "k1_walk", viewer: bool = False,
                 cam_size=(480, 640), walls=None, with_props: bool = True):
        prev = os.getcwd()
        os.chdir(DEPLOY_DIR)
        try:
            _load_tasks()
            from booster_deploy.utils.registry import get_task
            from booster_deploy.controllers.mujoco_controller import MujocoController

            class _Headless(MujocoController):
                def update_vel_command(self):  # we set commands directly
                    pass

            self.ctrl = _Headless(get_task(task))
        finally:
            os.chdir(prev)

        c = self.ctrl
        # Swap in the scene model (same robot + cameras + props), same init as booster_deploy.
        from booster_assets import BOOSTER_ASSETS_DIR

        xml = c.robot.cfg.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        c.mj_model = _build_scene_model(xml, walls=walls, with_props=with_props)
        c.mj_model.opt.timestep = c.cfg.mujoco.physics_dt
        c.mj_data = mujoco.MjData(c.mj_model)
        c.mj_data.qpos = np.concatenate([
            np.array(c.cfg.mujoco.init_pos, dtype=np.float32),
            np.array(c.cfg.mujoco.init_quat, dtype=np.float32),
            c.robot.default_joint_pos.numpy(),
        ])
        mujoco.mj_forward(c.mj_model, c.mj_data)

        self.dt = c.cfg.policy_dt  # 0.02 s (50 Hz policy, 500 Hz physics)
        self.default = c.robot.default_joint_pos.numpy().copy()
        self.overrides: dict[int, float] = {}  # joint index -> target (head, arm gestures)
        self._gesture = None  # (name, t0)
        self.lock = threading.Lock()
        self.viewer_enabled = viewer
        self.cam_h, self.cam_w = cam_size
        self._render_req: list = []
        self._thread = None
        self._running = False
        self.fallen = False

    # ---------- commands (thread-safe) ----------
    def set_velocity(self, vx: float, vy: float, vyaw: float):
        with self.lock:
            vc = self.ctrl.vel_command
            vc.lin_vel_x, vc.lin_vel_y, vc.ang_vel_yaw = float(vx), float(vy), float(vyaw)

    def set_head(self, yaw: float, pitch: float):
        with self.lock:
            self.overrides[HEAD_YAW] = float(np.clip(yaw, -1.0, 1.0))
            self.overrides[HEAD_PITCH] = float(np.clip(pitch, -0.3, 0.8))

    def wave(self, seconds: float = 3.0):
        with self.lock:
            self._gesture = ("wave", time.time(), seconds)

    def pose(self):
        """Return (x, y, yaw_rad, z) of the trunk."""
        with self.lock:
            q = self.ctrl.mj_data.qpos[:7].copy()
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), float(yaw), float(q[2])

    # ---------- camera ----------
    def render(self, camera: str = "head", timeout: float = 2.0) -> np.ndarray:
        """Render from 'head' (robot's eye) or 'chase'. Thread-safe; rendering
        happens on the sim thread so the GL context stays on one thread."""
        cam = {"head": "head_cam", "chase": "chase_cam"}.get(camera, camera)
        if not self._running:
            return self._render_now(cam)
        ev = threading.Event()
        box = {}
        with self.lock:
            self._render_req.append((cam, ev, box))
        ev.wait(timeout)
        return box.get("img")

    def _render_now(self, cam):
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.ctrl.mj_model, self.cam_h, self.cam_w)
        self._renderer.update_scene(self.ctrl.mj_data, camera=cam)
        return self._renderer.render().copy()

    # ---------- loop ----------
    def _apply_gesture(self, t):
        if self._gesture is None:
            for j in (R_SH_PITCH, R_SH_ROLL, R_EL_PITCH, R_EL_YAW):
                self.overrides.pop(j, None)
            return
        name, t0, dur = self._gesture
        el = t - t0
        if el > dur:
            self._gesture = None
            return
        if name == "wave":
            # Right arm up and out to the side, forearm swinging.
            self.overrides[R_SH_PITCH] = -0.3
            self.overrides[R_SH_ROLL] = -0.3
            self.overrides[R_EL_PITCH] = 0.0
            self.overrides[R_EL_YAW] = 1.0 + 0.5 * np.sin(2 * np.pi * 1.5 * el)

    def step(self):
        c = self.ctrl
        with self.lock:
            c.update_state()
            targets = c.policy_step()
            self._apply_gesture(time.time())
            for j, v in self.overrides.items():
                targets[j] = v
            c.ctrl_step(targets)
            reqs, self._render_req = self._render_req, []
            if c.mj_data.qpos[2] < 0.3:
                self.fallen = True
        for cam, ev, box in reqs:
            box["img"] = self._render_now(cam)
            ev.set()

    def _loop(self):
        viewer = None
        if self.viewer_enabled:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.ctrl.mj_model, self.ctrl.mj_data)
        nxt = time.perf_counter()
        while self._running and self.ctrl.is_running:
            self.step()
            if viewer is not None:
                if not viewer.is_running():
                    break
                with self.lock:
                    viewer.cam.lookat[:] = self.ctrl.mj_data.qpos[:3]
                    viewer.sync()
            nxt += self.dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
        self._running = False

    def start(self):
        self.ctrl.update_state()
        self.ctrl.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


if __name__ == "__main__":
    # Smoke test: stand, walk forward, turn, look, wave; save snapshots.
    from PIL import Image

    out = os.path.join(HERE, "runs")
    os.makedirs(out, exist_ok=True)
    sim = K1Sim(viewer="--viewer" in sys.argv).start()
    time.sleep(1.5)
    print("standing:", sim.pose())
    sim.set_velocity(0.5, 0, 0); time.sleep(4); sim.set_velocity(0, 0, 0)
    print("after walk 0.5 m/s x 4 s:", sim.pose())
    sim.set_velocity(0, 0, 0.6); time.sleep(2.5); sim.set_velocity(0, 0, 0)
    time.sleep(1)
    print("after turn:", sim.pose())
    sim.set_head(0.5, 0.2); time.sleep(1)
    Image.fromarray(sim.render("head")).save(f"{out}/smoke_head.png")
    sim.wave(2.5); time.sleep(1.2)
    Image.fromarray(sim.render("chase")).save(f"{out}/smoke_chase_wave.png")
    time.sleep(1.5)
    print("final:", sim.pose(), "fallen:", sim.fallen)
    print("saved", out)
    sim.close()
