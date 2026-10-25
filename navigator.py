"""Drop the K1 anywhere in the scanned house -> it works out where it is -> walks to a named place.

    .venv/bin/python navigator.py maps/agihouse --goal restroom                  # one trip from a random spawn
    .venv/bin/python navigator.py maps/agihouse --goal random --trials 20 --gif  # trials + success heatmap
    .venv/bin/python navigator.py maps/agihouse --selftest                       # localization only, no walking

No tags, no beacons. The robot has two position sources, as on the real K1:
  odometry   leg odometry (real: SDK rt/odometer_state). Smooth but drifts: here it is the sim's true
             motion corrupted by a scale error, a heading bias and a per-metre heading drift, starting
             at an unknown offset (the robot does not know where it was dropped).
  depth scan a horizontal slice of head-camera depth (real: K1 stereo depth). In the twin it is raycast
             against the scanned floor plan with range noise. Matched against the iPhone map it gives
             an absolute fix.
Global localization: turn in place 360 deg collecting scans, then score every reachable cell x 72 headings
against a likelihood field of the map's walls; refine the best peaks; confidence = best / runner-up.
While walking: every second a local scan match (+/-0.4 m, +/-8 deg around the odometry prediction)
corrects drift. Planning is A* on the cost map from the estimated pose. Gemma is kept out of the
steering loop; it confirms checkpoints / arrival on the real robot (conversation.py tools).
Ground truth is used only for scoring (did it arrive, localization error, collisions).
Outputs runs/<ts>-nav/: trials.json, heatmap.png (spawn points, green = arrived), trial_NN.png, gif.
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
import scan_map  # noqa: E402
from scan_map import RES  # noqa: E402

FOV = math.radians(90)      # usable horizontal FOV of the K1 head stereo depth (105 deg lens)
N_BEAMS = 45
MAX_RANGE = 5.0
SIGMA = 0.10                # m, likelihood field width
DROPOUT = 0.15              # fraction of beams with no depth return
UNSURE = 1.12              # below this best/runner-up ratio the robot asks the guest where it is
AMBIGUOUS = 1.04           # best/runner-up match score below this -> move and look again
LOS_CLEARANCE = 0.45       # m, route shortcuts keep this far from walls (corners are where it grazes)
CLUTTER = 0.10              # fraction of beams hitting something that is not in the map


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    """SE2 a * b, poses as (x, y, th)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


# ---------------------------------------------------------------- map model
class Field:
    """Likelihood field of the map's obstacles + raycaster (the twin's depth sensor)."""

    def __init__(self, m: scan_map.Map):
        from scipy import ndimage

        self.m = m
        d = ndimage.distance_transform_edt(~m.obstacle) * RES
        self.like = np.exp(-0.5 * (d / SIGMA) ** 2).astype(np.float32)
        self.H, self.W = m.obstacle.shape

    def lookup(self, xy):
        c = ((xy[..., 0] - self.m.ox) / RES).astype(np.int32)
        r = ((xy[..., 1] - self.m.oy) / RES).astype(np.int32)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        out = np.zeros(xy.shape[:-1], np.float32)
        out[ok] = self.like[r[ok], c[ok]]
        return out

    def raycast(self, pose, rng=None, fov=FOV, n=N_BEAMS, noise=0.04):
        """Ranges (m, nan = no return) for beams across the camera FOV at a map pose."""
        ang = pose[2] + np.linspace(-fov / 2, fov / 2, n)
        steps = np.arange(0.15, MAX_RANGE, RES * 0.8)
        xs = pose[0] + np.cos(ang)[:, None] * steps
        ys = pose[1] + np.sin(ang)[:, None] * steps
        c = ((xs - self.m.ox) / RES).astype(int)
        r = ((ys - self.m.oy) / RES).astype(int)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        hit = np.zeros_like(ok)
        hit[ok] = self.m.obstacle[r[ok], c[ok]]
        first = np.where(hit.any(1), hit.argmax(1), -1)
        out = np.where(first >= 0, steps[np.maximum(first, 0)], np.nan)
        if rng is not None:  # make it as messy as real stereo depth in a furnished, busy house
            out = out * (1 + noise * rng.standard_normal(n))
            out[rng.random(n) < DROPOUT] = np.nan                       # glass, dark or textureless surfaces
            short = rng.random(n) < CLUTTER                             # people / chairs not in the map
            out[short] = rng.uniform(0.4, 1.0, short.sum()) * np.nan_to_num(out[short], nan=MAX_RANGE)
        return out, np.linspace(-fov / 2, fov / 2, n)


def scan_points(ranges, angles, pose):
    """Scan (robot frame) -> points in the frame of `pose`."""
    ok = np.isfinite(ranges)
    a = angles[ok] + pose[2]
    return np.column_stack([pose[0] + ranges[ok] * np.cos(a), pose[1] + ranges[ok] * np.sin(a)])


def local_match(field, pts_robot, guess, dxy=0.4, dth=math.radians(8), step=0.05, thstep=math.radians(1)):
    """Best map pose near `guess` for points given in the robot frame. Returns (pose, score)."""
    if len(pts_robot) < 8:
        return guess, 0.0
    best, best_s = guess, -1.0
    for th in np.arange(-dth, dth + 1e-9, thstep * 2):
        a = guess[2] + th
        c, s = math.cos(a), math.sin(a)
        rot = pts_robot @ np.array([[c, s], [-s, c]])
        offs = np.arange(-dxy, dxy + 1e-9, step)
        ox, oy = np.meshgrid(offs, offs)
        cand = np.column_stack([guess[0] + ox.ravel(), guess[1] + oy.ravel()])
        sc = field.lookup(cand[:, None, :] + rot[None]).mean(1)
        i = int(np.argmax(sc))
        if sc[i] > best_s:
            best_s, best = float(sc[i]), (float(cand[i, 0]), float(cand[i, 1]), wrap(a))
    # fine pass
    g = best
    for th in np.arange(-math.radians(2), math.radians(2) + 1e-9, math.radians(0.5)):
        a = g[2] + th
        c, s = math.cos(a), math.sin(a)
        rot = pts_robot @ np.array([[c, s], [-s, c]])
        offs = np.arange(-0.06, 0.06 + 1e-9, 0.02)
        ox, oy = np.meshgrid(offs, offs)
        cand = np.column_stack([g[0] + ox.ravel(), g[1] + oy.ravel()])
        sc = field.lookup(cand[:, None, :] + rot[None]).mean(1)
        i = int(np.argmax(sc))
        if sc[i] > best_s:
            best_s, best = float(sc[i]), (float(cand[i, 0]), float(cand[i, 1]), wrap(a))
    return best, best_s


def global_localize(field, reach, pts_robot, cell=0.2, nth=72, hypotheses=False):
    """Score every reachable cell x heading. Returns (pose, score, runner_up_score), or with
    hypotheses=True the refined distinct peaks [(score, pose), ...] best first."""
    m = field.m
    k = max(1, int(round(cell / RES)))
    rr, cc = np.nonzero(reach[::k, ::k])
    cand = np.column_stack([m.ox + (cc * k + 0.5) * RES, m.oy + (rr * k + 0.5) * RES]).astype(np.float32)
    rng = np.random.default_rng(0)
    pts = pts_robot if len(pts_robot) <= 160 else pts_robot[rng.choice(len(pts_robot), 160, replace=False)]
    scores = np.zeros((nth, len(cand)), np.float32)
    for j, a in enumerate(np.linspace(-math.pi, math.pi, nth, endpoint=False)):
        c, s = math.cos(a), math.sin(a)
        rot = (pts @ np.array([[c, s], [-s, c]])).astype(np.float32)
        for i0 in range(0, len(cand), 4000):
            scores[j, i0:i0 + 4000] = field.lookup(cand[i0:i0 + 4000, None, :] + rot[None]).mean(1)
    # refine the top distinct peaks
    flat = np.argsort(scores.ravel())[::-1]
    peaks = []
    for f in flat[:400]:
        j, i = divmod(int(f), len(cand))
        p = (float(cand[i, 0]), float(cand[i, 1]), float(np.linspace(-math.pi, math.pi, nth, endpoint=False)[j]))
        if all(math.dist(p[:2], q[:2]) > 1.0 for q in peaks):
            peaks.append(p)
        if len(peaks) == 8:
            break
    refined = sorted((local_match(field, pts_robot, p, dxy=0.2, dth=math.radians(5))[::-1] for p in peaks),
                     key=lambda t: -t[0])
    if hypotheses:
        return refined
    best_s, best = refined[0]
    runner = next((s for s, p in refined[1:] if math.dist(p[:2], best[:2]) > 1.0), 0.0)
    return best, best_s, runner


# ---------------------------------------------------------------- the robot (sim backend)
class SimRobot:
    """K1 in the MuJoCo twin + drifting odometry + raycast depth slice."""

    def __init__(self, m, field, spawn, rng, gif=False):
        from k1_sim import K1Sim

        self.m, self.field, self.rng = m, field, rng
        m.places["start"] = {"xy": [spawn[0], spawn[1]], "yaw_deg": math.degrees(spawn[2])}
        self.sim = K1Sim(walls=scan_map.sim_walls(m), with_props=False)
        self.sim.ctrl.update_state()
        self.sim.ctrl.start()
        self.dt = self.sim.dt
        self.t = 0.0
        # odometry error model
        self.k_scale = 1 + rng.normal(0, 0.03)
        self.k_rot = 1 + rng.normal(0, 0.005)  # heading rate comes from the IMU gyro: good to ~0.5%
        self.drift = rng.normal(0, math.radians(1.5))  # rad per metre
        self.odom = (0.0, 0.0, 0.0)  # robot believes it starts at its own origin
        self._last = self._sim_pose()
        self.frames, self.frame_meta, self.gif = [], [], gif
        for _ in range(int(1.5 / self.dt)):
            self.step()

    def _sim_pose(self):
        x, y, yaw, _ = self.sim.pose()
        return (x, y, yaw)

    def true_pose(self):
        x, y, yaw = self._sim_pose()
        return self.m.to_map(x, y, yaw)

    def step(self, n=1):
        for _ in range(n):
            self.sim.step()
            self.t += self.dt
            cur = self._sim_pose()
            d = compose(inverse(self._last), cur)  # true motion in the robot frame
            dist = math.hypot(d[0], d[1])
            noisy = (d[0] * self.k_scale, d[1] * self.k_scale, d[2] * self.k_rot + self.drift * dist)
            self.odom = compose(self.odom, noisy)
            self._last = cur
            if self.gif and int(self.t / self.dt) % int(0.25 / self.dt) == 0:
                self.frames.append(self.sim.render("chase"))
                self.frame_meta.append((self.t, self.true_pose()))

    def drive(self, vx, vyaw):
        self.sim.set_velocity(vx, 0, vyaw)

    def scan(self):
        return self.field.raycast(self.true_pose(), self.rng)

    def head_image(self):
        return self.sim.render("head")

    def see_landmarks(self):
        """What Gemma would report from the head camera: [(kind, bearing_rad)] in the robot frame.
        Sim stand-in: landmarks that are in view, <= 4.5 m and not occluded, each reported with
        70% probability (+/-5 deg bearing noise), plus a 5% chance of one wrong label per frame.
        Real robot: landmarks.py's Gemma prompt on the head-camera frame (box centre -> bearing)."""
        tp, out = self.true_pose(), []
        kinds = sorted({p["landmark"] for p in self.m.places.values() if p.get("kind") == "landmark"})
        for p in self.m.places.values():
            if p.get("kind") != "landmark":
                continue
            ox, oy = p["object_xy"]
            r = math.hypot(ox - tp[0], oy - tp[1])
            b = wrap(math.atan2(oy - tp[1], ox - tp[0]) - tp[2])
            if r > 4.5 or abs(b) > FOV / 2 or self.rng.random() > 0.7:
                continue
            ray, _ = self.field.raycast((tp[0], tp[1], tp[2] + b), n=1, fov=0.0)
            if np.isfinite(ray[0]) and ray[0] < r - 0.5:  # a wall in between
                continue
            out.append((p["landmark"], b + math.radians(self.rng.normal(0, 5))))
        if self.rng.random() < 0.05:
            out.append((str(self.rng.choice(kinds)), float(self.rng.uniform(-FOV / 2, FOV / 2))))
        return out

    @property
    def fallen(self):
        return self.sim.fallen

    def close(self):
        self.sim.set_velocity(0, 0, 0)


# ---------------------------------------------------------------- navigation
class Navigator:
    def __init__(self, m, field, reach, log=print, dash=None):
        self.m, self.field, self.reach, self.log, self.dash = m, field, reach, log, dash
        self.T = None  # map <- odom correction
        self.events = []
        self.truth = True  # sim: score with ground truth; real robot: belief only
        from scipy import ndimage

        clear = ndimage.distance_transform_edt(~(m.obstacle | ~m.free)) * RES
        self.los = m.trav & (clear >= LOS_CLEARANCE)
        self.clear = clear
        # scoring: touching a mapped obstacle, or wandering >0.3 m off the mapped floor (unmapped != wall)
        self.collide = scan_map._dilate(m.obstacle, int(round(scan_map.BODY_RADIUS / RES))) | \
            ~scan_map._dilate(m.free, int(round(0.3 / RES)))

    def emit(self, kind, **kw):
        self.events.append({"t": round(self.robot.t, 2), "event": kind, **kw})
        self.log(f"  [{self.robot.t:6.1f}s] {kind} " + " ".join(f"{k}={v}" for k, v in kw.items()))
        if self.dash:
            self.dash.event(self.robot.t, kind, **kw)

    def estimate(self):
        return compose(self.T, self.robot.odom)

    def localize(self, robot, max_moves=2):
        """Turn in place collecting depth slices -> global scan match -> candidate poses (similar rooms
        look alike). Gemma's landmark sightings vote between near-ties. If the result is still ambiguous,
        step ~1 m into open space (chosen from depth alone) and look again; landmark sightings from every
        look keep voting. If still unsure afterwards the caller asks the guest (relocalize_near)."""
        self.robot, self.sightings = robot, []
        for attempt in range(max_moves + 1):
            pts_robot = scan_points_to_robot(self._spin_scan(), robot.odom)  # latest look only: no smear
            o = robot.odom
            t0 = time.time()
            hyps = global_localize(self.field, self.reach, pts_robot, hypotheses=True)
            (s, pose), runner, picked = self._pick(hyps)
            conf = s / max(runner, 1e-6)
            if self.dash:
                self.dash.hypotheses(robot.t, hyps, chosen=pose)
            if picked is not None:
                self.emit("landmarks_vote", seen=picked["seen"], agree=picked["agree"], hypotheses=len(hyps),
                          chose_rank=picked["rank"])
                conf = max(conf, AMBIGUOUS) if picked["clear"] else conf
            done = conf >= AMBIGUOUS or attempt == max_moves
            self.emit("localized" if done else "ambiguous", score=round(s, 3), ratio=round(conf, 2),
                      search_s=round(time.time() - t0, 1), est=tuple(round(float(v), 2) for v in pose))
            if done or not self._step_into_open(self._last_cloud):
                break
        self.T = compose(pose, inverse(o))
        self.confidence = conf
        return self.estimate(), s, conf

    def relocalize_near(self, xys, radius=6.0):
        """The guest told us roughly where we are ("we're by the kitchen"): search again, only within
        `radius` of that spot, using the look-around scan we already have (no extra motion)."""
        rr, cc = np.nonzero(self.reach)
        xs, ys = self.m.ox + (cc + 0.5) * RES, self.m.oy + (rr + 0.5) * RES
        xys = [xys] if np.ndim(xys) == 1 else list(xys)  # several couches -> search near all of them
        mask = np.zeros_like(self.reach)
        near = np.min([np.hypot(xs - x, ys - y) for x, y in xys], axis=0) <= radius
        mask[rr[near], cc[near]] = True
        o = self._cloud_odom
        hyps = global_localize(self.field, mask, scan_points_to_robot(self._last_cloud, o), hypotheses=True)
        (s, pose), runner, _ = self._pick(hyps)
        self.T = compose(pose, inverse(o))
        self.confidence = s / max(runner, 1e-6)
        est = self.estimate()
        self.emit("localized_with_hint", near=len(xys), score=round(s, 3),
                  est=tuple(round(float(v), 2) for v in est))
        return est

    def _pick(self, hyps, near=0.05):
        """Among scan hypotheses within `near` of the best score, choose the one whose predicted
        landmarks match what Gemma saw during the spin. Returns ((score, pose), runner_up, info)."""
        best_s = hyps[0][0]
        cands = [(i, s, p) for i, (s, p) in enumerate(hyps) if s >= best_s * (1 - near)]
        runner_all = next((s for s, p in hyps[1:] if math.dist(p[:2], hyps[0][1][:2]) > 1.0), 0.0)
        if len(cands) < 2 or not self.sightings:
            return hyps[0], runner_all, None
        o_now = self.robot.odom
        lms = [(p["landmark"], p["object_xy"]) for p in self.m.places.values() if p.get("kind") == "landmark"]

        def agree(pose):
            n = 0
            for kind, bearing, o_obs in self.sightings:
                v = compose(compose(pose, inverse(o_now)), o_obs)  # where the robot was when it saw it
                for k2, (ox, oy) in lms:
                    if k2 == kind and math.hypot(ox - v[0], oy - v[1]) < 5.5 and \
                            abs(wrap(math.atan2(oy - v[1], ox - v[0]) - v[2] - bearing)) < math.radians(25):
                        n += 1
                        break
            return n

        votes = sorted(((agree(p), s, i, p) for i, s, p in cands), key=lambda t: (-t[0], -t[1]))
        a0, s0, i0, p0 = votes[0]
        a1 = votes[1][0]
        runner = next((s for a, s, i, p in votes[1:] if math.dist(p[:2], p0[:2]) > 1.0), runner_all)
        return (s0, p0), runner, {"seen": len(self.sightings), "agree": a0, "rank": i0, "clear": a0 > a1}

    def _step_into_open(self, cloud, dist=1.0, on_step=None):
        """Walk ~dist m toward the direction with the most free space (from depth only)."""
        robot, o = self.robot, self.robot.odom
        d = cloud - np.array(o[:2])
        ang, rng_ = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
        bins = np.linspace(-math.pi, math.pi, 25)
        free = [np.percentile(rng_[(ang >= a) & (ang < b)], 20) if ((ang >= a) & (ang < b)).sum() > 3 else 0
                for a, b in zip(bins, bins[1:])]
        free = np.array(free)
        free = np.minimum(free, np.minimum(np.roll(free, 1), np.roll(free, -1)))  # clear for the whole body width
        k = int(np.argmax(free))
        if free[k] < dist + 1.2:
            return False
        heading = (bins[k] + bins[k + 1]) / 2
        self.emit("repositioning", heading_deg=round(math.degrees(heading)), clear_m=round(float(free[k]), 1))
        t_end = robot.t + 25
        start = robot.odom[:2]
        while robot.t < t_end and math.dist(robot.odom[:2], start) < dist:
            err = wrap(heading - robot.odom[2])
            robot.drive(0.0 if abs(err) > 0.3 else 0.3, float(np.clip(1.5 * err, -0.8, 0.8)))
            robot.step()
            if on_step:
                on_step()
        robot.drive(0, 0)
        robot.step(int(0.5 / robot.dt))
        return True

    def _spin_scan(self):
        robot = self.robot
        pts, turned, last_th = [], 0.0, robot.odom[2]
        robot.drive(0, 0.6)
        next_scan, t_end = robot.t, robot.t + 30
        while turned < 2 * math.pi + 0.2 and robot.t < t_end:  # one full turn, by the robot's own heading
            if robot.t >= next_scan:
                r, a = robot.scan()
                pts.append(scan_points(r, a, robot.odom))
                if hasattr(robot, "see_landmarks") and len(pts) % 3 == 0:  # Gemma: ~1 frame / 1.2 s
                    seen = robot.see_landmarks()
                    self.sightings += [(k, b, robot.odom) for k, b in seen]
                    if self.dash:
                        self.dash.sightings(robot.t, robot.odom, seen)  # odom frame until localized
                next_scan += 0.4
            robot.step()
            turned += abs(wrap(robot.odom[2] - last_th))
            last_th = robot.odom[2]
        robot.drive(0, 0)
        robot.step(int(0.8 / robot.dt))
        self._last_cloud, self._cloud_odom = np.vstack(pts), robot.odom
        return self._last_cloud

    def correct(self, wide=False):
        r, a = self.robot.scan()
        self._last_scan = (r, a)
        ok = np.isfinite(r)
        if ok.sum() < 10:
            return None
        pts = np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])
        guess = self.estimate()
        pose, s = local_match(self.field, pts, guess, dxy=1.0, dth=math.radians(25)) if wide else \
            local_match(self.field, pts, guess, dth=math.radians(12))
        _, s0 = local_match(self.field, pts, guess, dxy=0.0, dth=0.0)
        if s > 0.35 and s > s0 + 0.02:
            self.T = compose(pose, inverse(self.robot.odom))
            return math.dist(pose[:2], guess[:2])
        return 0.0

    def go_to(self, goal, max_v=0.4, max_w=0.8, t_budget=None):
        robot = self.robot
        est = self.estimate()
        path, msg = scan_map.plan(self.m, goal, start_xy=est[:2], los_mask=self.los, clearance=self.clear)
        if path is None:
            self.emit("plan_failed", reason=msg)
            return {"ok": False, "result": msg, "path": [], "traj_true": [], "traj_est": [], "fixes": 0, "mean_fix_m": 0.0, "path_m": 0.0}
        self._path, self._wp = path, 1  # for commentary (escort.py)
        self.emit("planned", route=msg)
        if self.dash:
            self.dash.plan(robot.t, path)
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        t_end = robot.t + (t_budget or 30 + 3 * length / 0.3)
        final_checks = 0
        i, next_fix, traj_true, traj_est, fixes = 1, robot.t + 1.0, [], [], []
        while True:
            est = self.estimate()
            tp = robot.true_pose() if self.truth else est
            if self.truth:
                traj_true.append(tp[:2])
            traj_est.append(est[:2])
            r, c = self.m.cell(*tp[:2])
            if robot.fallen:
                return self._done(False, "fell", path, traj_true, traj_est, fixes)
            if self.truth and (not self.m.inside(r, c) or self.collide[r, c]):
                return self._done(False, f"touched an obstacle at ({tp[0]:.1f}, {tp[1]:.1f})", path, traj_true,
                                  traj_est, fixes)
            if robot.t > t_end:
                return self._done(False, "timed out", path, traj_true, traj_est, fixes)
            if self.dash and robot.t >= getattr(self, "_next_dash", 0):
                self._next_dash = robot.t + 0.25
                self.dash.pose(robot.t, est, tp, getattr(self, "_last_scan", None))
                self.dash.metric(robot.t, "distance_to_goal_m", math.dist(est[:2], path[-1]))
                if robot.t >= getattr(self, "_next_cam", 0) and hasattr(robot, "head_image"):
                    self._next_cam = robot.t + 1.0
                    self.dash.camera(robot.t, robot.head_image())
            if robot.t >= next_fix:
                d = self.correct()
                if d:
                    fixes.append(d)
                next_fix = robot.t + 1.0
            tx, ty = path[i]
            self._wp = i
            dist = math.hypot(tx - est[0], ty - est[1])
            last = i == len(path) - 1
            if dist < (0.35 if last else 0.2):
                if last:
                    robot.drive(0, 0)
                    robot.step(int(0.8 / robot.dt))
                    if final_checks < 1:  # standing still: re-fix against the map before declaring arrival
                        final_checks += 1
                        # corridors fix only the across-track position; a full look-around fixes along-track too
                        cloud = scan_points_to_robot(self._spin_scan(), robot.odom)
                        cur = self.estimate()
                        pose, sc = local_match(self.field, cloud, cur, dxy=1.2, dth=math.radians(20))
                        _, sc0 = local_match(self.field, cloud, cur, dxy=0.0, dth=0.0)
                        if sc > 0.35 and sc > sc0 + 0.03 and math.dist(pose[:2], cur[:2]) < 0.6:
                            self.T = compose(pose, inverse(robot.odom))
                        est = self.estimate()
                        if math.hypot(tx - est[0], ty - est[1]) > 0.35:
                            self.emit("final_fix", moved_m=round(math.hypot(tx - est[0], ty - est[1]), 2))
                            continue
                    tp = robot.true_pose() if self.truth else self.estimate()
                    err = math.dist(tp[:2], self.m.place(goal))
                    ok = err < 0.75
                    self.emit("arrived" if ok else "stopped_short", goal=goal,
                              **{"true_error_m" if self.truth else "est_error_m": round(err, 2)})
                    return self._done(ok, "arrived" if ok else f"stopped {err:.2f} m from goal", path,
                                      traj_true, traj_est, fixes)
                i += 1
                self.emit("waypoint", n=i - 1, of=len(path) - 1)
                continue
            err = wrap(math.atan2(ty - est[1], tx - est[0]) - est[2])
            if abs(err) > 0.35:
                robot.drive(0, float(np.clip(1.5 * err, -max_w, max_w)))
            else:
                robot.drive(float(np.clip(dist, 0.2, max_v)), float(np.clip(1.2 * err, -max_w, max_w)))
            robot.step()

    def _done(self, ok, reason, path, traj_true, traj_est, fixes):
        self.robot.drive(0, 0)
        return {"ok": ok, "result": reason, "path": path, "traj_true": traj_true, "traj_est": traj_est,
                "fixes": len(fixes), "mean_fix_m": round(float(np.mean(fixes)), 3) if fixes else 0.0,
                "path_m": round(sum(math.dist(a, b) for a, b in zip(path, path[1:])), 2)}


def scan_points_to_robot(pts, o):
    """Points in the odom frame -> frame of odom pose o."""
    c, s = math.cos(o[2]), math.sin(o[2])
    d = pts - np.array(o[:2])
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


# ---------------------------------------------------------------- trials
def sample_spawn(m, field, reach, rng, clearance=0.5):
    from scipy import ndimage

    d = ndimage.distance_transform_edt(~(m.obstacle | ~m.free)) * RES
    rr, cc = np.nonzero(reach & (d >= clearance))
    i = rng.integers(len(rr))
    x, y = m.xy(rr[i], cc[i])
    return (x, y, float(rng.uniform(-math.pi, math.pi)))


def plot_trial(m, res, spawn, est0, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = _base(m, plt)
    if res.get("path"):
        ax.plot(*zip(*res["path"]), "-o", color="#e08a1e", lw=2, ms=3, label="plan (from estimate)")
    if res.get("traj_true"):
        ax.plot(*zip(*res["traj_true"]), "-", color="#7b3fb5", lw=1.8, label="K1 true path")
    if res.get("traj_est"):
        ax.plot(*zip(*res["traj_est"]), "--", color="#2a9d8f", lw=1.2, label="K1 belief")
    ax.plot(spawn[0], spawn[1], "*", color="#d0453a", ms=16, label="dropped here")
    ax.arrow(spawn[0], spawn[1], 0.6 * math.cos(spawn[2]), 0.6 * math.sin(spawn[2]), width=0.05, color="#d0453a")
    ax.plot(est0[0], est0[1], "x", color="#2a9d8f", ms=12, mew=3, label="localized")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{res.get('goal', '')}: {res['result']}")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _base(m, plt):
    H, W = m.free.shape
    img = np.full((H, W, 3), 0.93)
    img[m.free] = (0.99, 0.985, 0.97)
    img[m.obstacle] = (0.22, 0.24, 0.3)
    ext = [m.ox, m.ox + W * RES, m.oy, m.oy + H * RES]
    fig, ax = plt.subplots(figsize=(11, 11 * H / W))
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    for name, p in m.places.items():
        if p.get("kind") == "landmark":
            ax.plot(*p["xy"], "o", color="#888", ms=4)
            ax.annotate(name.replace("_", " "), p["xy"], xytext=(4, 3), textcoords="offset points", fontsize=8,
                        color="#555")
    ax.set_axis_off()
    return fig, ax


def plot_heatmap(m, trials, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = _base(m, plt)
    for t in trials:
        s, g = t["spawn"], m.place(t["goal"])
        col = "#2a9d4f" if t["ok"] else "#d0453a"
        ax.annotate("", g, s[:2], arrowprops=dict(arrowstyle="->", color=col, alpha=0.35, lw=1))
        ax.plot(s[0], s[1], "o", color=col, ms=10, mec="white", mew=1.5)
    n, ok = len(trials), sum(t["ok"] for t in trials)
    ax.set_title(f"Random drops in the AGI House twin: {ok}/{n} arrived "
                 f"(green = arrived, red = failed; arrow = goal)", fontsize=12)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _dash(a, m):
    if not (a.rerun or a.rerun_save):
        return None
    from dashboard import Dash

    return Dash(m, save=a.rerun_save)


def run_real(a, m, field, reach, out, start_yaw):
    """One trip on the real K1: no ground truth, arrival judged on the estimated pose."""
    from real_robot import RealRobot

    if a.goal == "random" or a.goal not in m.places:
        sys.exit(f"--goal must be a place: {', '.join(m.places)}")
    robot = RealRobot(a.url)
    nav = Navigator(m, field, reach, dash=_dash(a, m))
    nav.robot, nav.truth = robot, False
    t0, ratio, res = time.time(), None, {"ok": False, "result": "interrupted"}
    try:
        if a.start_at:
            x, y = m.place(a.start_at)
            spawn = (x, y, math.radians(a.yaw if a.yaw is not None else
                                        m.places[a.start_at].get("yaw_deg", start_yaw)))
            nav.T = compose(spawn, inverse(robot.odom))
            nav.emit("start_given", at=a.start_at, yaw_deg=round(math.degrees(spawn[2])))
            nav.emit("start_fix", moved_m=round(nav.correct() or 0.0, 2))
        else:
            spawn, s, ratio = nav.localize(robot)
        est0 = nav.estimate()
        print(f"real: start estimate ({est0[0]:.2f}, {est0[1]:.2f}, {math.degrees(est0[2]):.0f} deg) -> {a.goal}")
        res = nav.go_to(a.goal)
    except KeyboardInterrupt:
        print("interrupted: stopping")
        est0 = spawn = nav.estimate() if nav.T else (0, 0, 0)
    finally:
        robot.close()
    res.update(goal=a.goal, spawn=list(spawn), loc_ratio=ratio and round(ratio, 2), wall_s=round(time.time() - t0, 1),
               events=nav.events)
    plot_trial(m, res, spawn, est0, os.path.join(out, "trial_00.png"))
    json.dump([{k: v for k, v in res.items() if k not in ("traj_true", "traj_est", "path")}],
              open(os.path.join(out, "trials.json"), "w"), indent=1)
    print(f"real: {res['result']} ({res.get('fixes', 0)} drift fixes, {res['wall_s']} s) -> {os.path.relpath(out, HERE)}")
    sys.stdout.flush(); os._exit(0 if res["ok"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--goal", default="random")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", action="store_true", help="chase-cam GIF of the first trial")
    ap.add_argument("--selftest", action="store_true", help="localization only: N random drops, no walking")
    ap.add_argument("--hint", action="store_true",
                    help="when unsure, simulate asking the guest (they name the nearest landmark)")
    ap.add_argument("--rerun", action="store_true", help="live Rerun dashboard (first trial)")
    ap.add_argument("--rerun-save", help="record the dashboard to this .rrd instead")
    ap.add_argument("--robot", default="sim", choices=["sim", "real"], help="real = K1 behind robot_server.py")
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="robot_server.py on the K1's Jetson")
    ap.add_argument("--start-at", help="real: known start place (skips the global search)")
    ap.add_argument("--yaw", type=float, help="real: heading at --start-at, deg in map frame")
    a = ap.parse_args()

    from landmarks import reachable

    m = scan_map.Map(a.map_dir)
    start_yaw = m.places.pop("start", {}).get("yaw_deg", 0.0)
    field = Field(m)
    reach = reachable(scan_map.Map(a.map_dir))  # floor connected to the entrance
    goals = [k for k, v in m.places.items() if v.get("kind") == "landmark"]
    rng = np.random.default_rng(a.seed)
    out = os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S") + f"-nav-s{a.seed}" + ("-hint" if a.hint else "") + f"-{os.getpid()}")
    os.makedirs(out, exist_ok=True)

    if a.robot == "real":
        run_real(a, m, field, reach, out, start_yaw)
    if a.selftest:
        errs = []
        for k in range(a.trials):
            spawn = sample_spawn(m, field, reach, rng)
            pts = []
            for th in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                pose = (spawn[0], spawn[1], spawn[2] + th)
                r, an = field.raycast(pose, rng)
                pts.append(scan_points(r, an, (0, 0, th)))
            pose, s, runner = global_localize(field, reach, np.vstack(pts))
            e = math.dist(pose[:2], spawn[:2])
            eth = abs(math.degrees(wrap(pose[2] - spawn[2])))
            errs.append((e, eth))
            print(f"  drop {k}: error {e:.2f} m / {eth:.1f} deg  score {s:.2f}  ratio {s / max(runner, 1e-6):.2f}")
        e = np.array(errs)
        ok = (e[:, 0] < 0.5) & (e[:, 1] < 10)
        print(f"selftest: {ok.sum()}/{len(e)} within 0.5 m / 10 deg; median {np.median(e[:, 0]):.2f} m, "
              f"{np.median(e[:, 1]):.1f} deg")
        sys.stdout.flush(); os._exit(0)

    trials = []
    for k in range(a.trials):
        goal = a.goal if a.goal != "random" else str(rng.choice(goals))
        spawn = sample_spawn(m, field, reach, rng)
        print(f"trial {k}: dropped at ({spawn[0]:.1f}, {spawn[1]:.1f}, {math.degrees(spawn[2]):.0f} deg) -> {goal}")
        robot = SimRobot(m, field, spawn, np.random.default_rng(a.seed * 1000 + k), gif=a.gif and k == 0)
        nav = Navigator(m, field, reach, dash=_dash(a, m) if k == 0 else None)
        t0 = time.time()
        est0, s, ratio = nav.localize(robot)
        asked = False
        if a.hint and ratio < UNSURE:  # "Which room are we in?" -> the guest names the nearest landmark
            tp = robot.true_pose()
            hint = min((math.dist(tp[:2], p["object_xy"]), p["object_xy"]) for p in m.places.values()
                       if p.get("kind") == "landmark")[1]
            est0, asked = nav.relocalize_near(hint), True
        tp = robot.true_pose()
        loc_err = math.dist(est0[:2], tp[:2])
        loc_th = abs(math.degrees(wrap(est0[2] - tp[2])))
        res = nav.go_to(goal)
        res.update(goal=goal, spawn=list(spawn), loc_err_m=round(loc_err, 2), loc_err_deg=round(loc_th, 1),
                   loc_ratio=round(ratio, 2), asked_guest=asked, sim_time_s=round(robot.t, 1), wall_s=round(time.time() - t0, 1),
                   events=nav.events)
        print(f"  -> {res['result']}  (localization error {loc_err:.2f} m / {loc_th:.1f} deg, "
              f"{res['fixes']} drift fixes, sim {robot.t:.0f}s)")
        plot_trial(m, res, spawn, est0, os.path.join(out, f"trial_{k:02d}.png"))
        if robot.frames:
            from PIL import Image

            ims = [Image.fromarray(f).resize((480, 360)) for f in robot.frames]
            ims[0].save(os.path.join(out, "trial_00.gif"), save_all=True, append_images=ims[1:], duration=125,
                        loop=0)
        robot.close()
        trials.append({k2: v for k2, v in res.items() if k2 not in ("traj_true", "traj_est", "path")})
        json.dump(trials, open(os.path.join(out, "trials.json"), "w"), indent=1)
        plot_heatmap(m, trials, os.path.join(out, "heatmap.png"))
    n, ok = len(trials), sum(t["ok"] for t in trials)
    print(f"\n{ok}/{n} arrived; median localization error "
          f"{np.median([t['loc_err_m'] for t in trials]):.2f} m -> {os.path.relpath(out, HERE)}")
    sys.stdout.flush(); os._exit(0)  # skip MuJoCo renderer teardown


if __name__ == "__main__":
    main()
