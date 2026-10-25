"""Live Rerun dashboard for the navigator: cost map, plan, belief vs truth, depth slice, Gemma, events.

    .venv/bin/python navigator.py maps/agihouse --goal restroom --rerun            # opens the viewer
    .venv/bin/python navigator.py maps/agihouse --goal restroom --rerun-save x.rrd # record for the video

Everything is logged in map coordinates (metres) on the `sim` timeline:
  map/costmap        floor plan (walls, inflation, unknown) as an image placed in metric space
  map/landmarks      Gemma landmarks from the walkthrough
  map/plan           A* route        map/belief  where the robot thinks it is (+ trail)
  map/truth          ground truth (sim only)      map/scan  current depth slice, at the belief pose
  map/hypotheses     global-localization candidates, map/gemma  landmark sightings
  camera/head        head camera    log/events  navigator events (also spoken by conversation.py)
"""
from __future__ import annotations

import math

import numpy as np


class Dash:
    def __init__(self, m, spawn_viewer=True, save=None, app="k1_escort"):
        import rerun as rr

        self.rr, self.m = rr, m
        rr.init(app)
        if save:
            rr.save(save)
        elif spawn_viewer:
            rr.spawn()
        self.trail_b, self.trail_t = [], []
        self._log_static()

    def _time(self, t):
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("sim", duration=t)
        else:
            rr.set_time_seconds("sim", t)

    def _log_static(self):
        rr, m = self.rr, self.m
        H, W = m.free.shape
        img = np.full((H, W, 3), 140, np.uint8)
        img[m.free] = (245, 245, 240)
        img[m.free & ~m.trav] = (215, 222, 240)
        img[m.obstacle] = (50, 52, 64)
        res = 0.05
        # image row 0 is the top -> flip so +y is up, then place it in metres with a 2D transform
        rr.log("map/costmap", rr.Image(img[::-1]), rr.Transform3D(
            translation=[m.ox, m.oy + H * res, 0], mat3x3=[[res, 0, 0], [0, -res, 0], [0, 0, 1]]), static=True)
        names, pts = [], []
        for k, p in m.places.items():
            if p.get("kind") == "landmark":
                names.append(k.replace("_", " "))
                pts.append(p["object_xy"])
        if pts:
            rr.log("map/landmarks", rr.Points2D(pts, labels=names, radii=0.15, colors=[255, 170, 40]), static=True)

    def plan(self, t, path):
        self._time(t)
        self.rr.log("map/plan", self.rr.LineStrips2D([path], colors=[224, 138, 30], radii=0.05))

    def hypotheses(self, t, hyps, chosen=None):
        self._time(t)
        pts = [p[:2] for _, p in hyps]
        labels = [f"{s:.2f}" for s, _ in hyps]
        cols = [[42, 157, 143] if chosen is not None and math.dist(p[:2], chosen[:2]) < 0.3 else [150, 150, 150]
                for _, p in hyps]
        self.rr.log("map/hypotheses", self.rr.Points2D(pts, labels=labels, colors=cols, radii=0.25))

    def pose(self, t, belief, truth=None, scan=None):
        rr = self.rr
        self._time(t)
        self.trail_b.append(belief[:2])
        rr.log("map/belief", rr.Arrows2D(origins=[belief[:2]], vectors=[[0.6 * math.cos(belief[2]),
                                                                        0.6 * math.sin(belief[2])]],
                                         colors=[42, 157, 143], radii=0.06))
        rr.log("map/belief_trail", rr.LineStrips2D([self.trail_b], colors=[42, 157, 143], radii=0.03))
        if truth is not None:
            self.trail_t.append(truth[:2])
            rr.log("map/truth", rr.Arrows2D(origins=[truth[:2]], vectors=[[0.6 * math.cos(truth[2]),
                                                                          0.6 * math.sin(truth[2])]],
                                            colors=[123, 63, 181], radii=0.05))
            rr.log("map/truth_trail", rr.LineStrips2D([self.trail_t], colors=[123, 63, 181], radii=0.02))
        if scan is not None:
            r, a = scan
            ok = np.isfinite(r)
            ang = a[ok] + belief[2]
            pts = np.column_stack([belief[0] + r[ok] * np.cos(ang), belief[1] + r[ok] * np.sin(ang)])
            rr.log("map/scan", rr.Points2D(pts, colors=[230, 57, 70], radii=0.05))

    def sightings(self, t, belief, seen):
        rr = self.rr
        self._time(t)
        if not seen:
            return
        vec = [[3.0 * math.cos(belief[2] + b), 3.0 * math.sin(belief[2] + b)] for _, b in seen]
        rr.log("map/gemma", rr.Arrows2D(origins=[belief[:2]] * len(seen), vectors=vec,
                                        labels=[k.replace("_", " ") for k, _ in seen],
                                        colors=[255, 200, 60], radii=0.02))

    def camera(self, t, img):
        if img is None:
            return
        self._time(t)
        self.rr.log("camera/head", self.rr.Image(img))

    def event(self, t, kind, **kw):
        self._time(t)
        self.rr.log("log/events", self.rr.TextLog(f"{kind} " + " ".join(f"{k}={v}" for k, v in kw.items())))

    def metric(self, t, name, value):
        self._time(t)
        rr = self.rr
        rr.log(f"metrics/{name}", rr.Scalars(value) if hasattr(rr, "Scalars") else rr.Scalar(value))
