"""The whole demo in one process: talk to the K1, it localizes, confirms, and walks you there narrating.

    .venv/bin/python escort.py maps/agihouse --text                      # type to the robot
    .venv/bin/python escort.py maps/agihouse                             # push-to-talk (Gemma hears audio)
    .venv/bin/python escort.py maps/agihouse --script "Hi, can you take me to the restroom?" "Yes please"
    .venv/bin/python escort.py maps/agihouse --text --rerun              # + live dashboard

System 2 = Gemma 4 E2B on LiteRT-LM (on-device): hears/reads the guest, sees the head camera, decides
(go_to / where_am_i / wave / stop ...), confirms before walking. System 1 = navigator.py: scan-matching
localization, A* on the cost map, drift-corrected path following, Booster's RL walking policy (MuJoCo twin).
Commentary is generated from navigator events (plan, turns, landmarks passed, arrival), so the robot talks
while it walks without waiting on the model. A typed `s` (or space-bar on the real robot) stops instantly.
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


YES = __import__("re").compile(r"\b(yes|yeah|yep|sure|ok|okay|please do|let'?s go|go ahead|alright)\b")


def pretty(name):
    base = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
    return {"tv": "TV"}.get(base, base.replace("_", " "))


class Voice:
    """Non-blocking speech so the robot keeps walking while it talks; throttled commentary."""

    def __init__(self, enabled=True, rate=165):
        self.enabled = enabled and platform.system() == "Darwin" and shutil.which("say")
        self.proc, self.rate, self.last = None, rate, 0.0
        self.clock, self.lines = None, []  # for --video: (sim time, who, text)

    def say(self, text, wait=False, urgent=False, min_gap=4.0):
        now = time.time()
        if not urgent and now - self.last < min_gap and self.busy():
            return False
        print(f"\n  K1: {text}", flush=True)
        self.last = now
        if self.clock:
            self.lines.append((self.clock(), "K1", text))
        if self.enabled:
            if urgent and self.busy():
                self.proc.terminate()
            elif self.busy():
                self.proc.wait()
            self.proc = subprocess.Popen(["say", "-r", str(self.rate), text])
            if wait:
                self.proc.wait()
        return True

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def wait(self):
        if self.busy():
            self.proc.wait()


class Commentator:
    """Turns navigator state into short spoken lines for an elderly guest."""

    def __init__(self, m, voice):
        self.m, self.voice, self.mentioned, self.last_turn = m, voice, set(), 0.0
        self.lms = {k: p for k, p in m.places.items() if p.get("kind") == "landmark"}

    def on_event(self, kind, kw, nav):
        v = self.voice
        if kind == "planned":
            self.said, self.mentioned = set(), set()  # fresh commentary for each trip
            metres = float(kw["route"].split(",")[1].split()[0])
            v.say(f"It's about {round(metres)} metres. Walk with me, I'll go slowly.", urgent=True)
        elif kind == "arrived":
            v.say(f"Here we are, the {pretty(kw['goal'])}. Take your time. Is there anything else?", urgent=True)
        elif kind == "stopped_short":
            v.say(f"We're nearly at the {pretty(kw['goal'])}, just a few steps ahead.", urgent=True)
        elif kind in ("touched", "plan_failed"):
            v.say("I need to stop for a moment and find another way.", urgent=True)

    def tick(self, nav, goal, path, i):
        """Called while walking: turns ahead, landmarks we pass, halfway."""
        est = nav.estimate()
        t = nav.robot.t
        remaining = math.dist(est[:2], path[i]) + sum(math.dist(a, b) for a, b in zip(path[i:], path[i + 1:]))
        total = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        if not hasattr(self, "said"):
            self.said = set()
        if total > 10 and "half" not in self.said and remaining < total / 2:
            self.said.add("half")
            if self.voice.say("We're about halfway there. You're doing well."):
                return
        if "near" not in self.said and 3.0 < remaining < 5.0:
            self.said.add("near")
            if self.voice.say(f"The {pretty(goal)} is just ahead."):
                return
        if i < len(path) - 1 and t - self.last_turn > 6:
            a, b, c = path[i - 1], path[i], path[i + 1]
            d = math.dist(est[:2], b)
            turn = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
            turn = (turn + 180) % 360 - 180
            if d < 1.2 and abs(turn) > 35 and math.dist(b, c) > 1.5:
                if self.voice.say(f"We'll turn {'left' if turn > 0 else 'right'} here."):
                    self.last_turn = t
                return
        for k, p in self.lms.items():
            kind = p["landmark"]
            if k == goal or kind in self.mentioned or kind in ("trash_can",) or \
                    kind == self.m.places.get(goal, {}).get("landmark"):
                continue
            ox, oy = p["object_xy"]
            if math.hypot(ox - est[0], oy - est[1]) < 2.2:
                side = math.atan2(oy - est[1], ox - est[0]) - est[2]
                side = (side + math.pi) % (2 * math.pi) - math.pi
                verb = "are" if pretty(k).endswith("s") else "is"
                if abs(side) < math.radians(150) and self.voice.say(
                        f"We're passing the {pretty(k)}, on your {'left' if side > 0 else 'right'}."):
                    self.mentioned.add(kind)
                return


class Body:
    """What conversation.py's tools/turn need from a robot, backed by the navigator's SimRobot."""

    def __init__(self, robot, voice):
        self.r, self.voice, self.log, self.b = robot, voice, [], type("B", (), {"name": "nav"})()

    def get_frame(self):
        return self.r.head_image()

    def status(self):
        return {"moving": False}

    def say(self, text):
        self.voice.say(text, urgent=True)

    def wave(self):
        (self.r.sim if hasattr(self.r, "sim") else self.r).wave()
        self.log.append("wave")
        return "waved"

    def look(self, yaw_deg, pitch_deg):
        (self.r.sim if hasattr(self.r, "sim") else self.r).set_head(math.radians(float(np.clip(yaw_deg, -55, 55))),
                            math.radians(float(np.clip(pitch_deg, -15, 45))))
        self.r.step(int(0.6 / self.r.dt))
        return f"looking yaw {yaw_deg:.0f}, pitch {pitch_deg:.0f}"

    def turn(self, degrees):
        return "I'll turn as part of walking; tell me where to go."

    def stop(self):
        self.r.drive(0, 0)
        return "stopped"


class NavTools:
    """go_to / where_am_i for Gemma; go_to only queues the trip so the reply is spoken first."""

    def __init__(self, nav, m):
        self.nav, self.m, self.pending, self.proposed, self.state = nav, m, None, None, "standing still"

    def distance(self, k):
        if self.nav.T is None:
            return "some way"
        import scan_map

        path, _ = scan_map.plan(self.m, k, start_xy=self.nav.estimate()[:2], los_mask=self.nav.los)
        return f"about {round(sum(math.dist(a, b) for a, b in zip(path, path[1:])))} metres" if path else "some way"

    def place_names(self):
        return [p for p, v in self.m.places.items() if p != "start" and v.get("kind") != "nogo"
                and not p.rsplit("_", 1)[-1].isdigit()]

    SYNONYMS = {  # spoken word -> landmark kind
        "front": "front_entrance", "entrance": "front_entrance", "door": "front_entrance", "doorway": "front_entrance",
        "restroom": "restroom", "bathroom": "restroom", "toilet": "restroom", "loo": "restroom", "washroom": "restroom",
        "kitchen": "kitchen", "sink": "kitchen", "fridge": "fridge", "refrigerator": "fridge",
        "cooktop": "cooktop", "stove": "cooktop", "oven": "cooktop", "bar": "bar", "drinks": "bar",
        "stairs": "stairs", "staircase": "stairs", "steps": "stairs", "fireplace": "fireplace",
        "couch": "couch", "sofa": "couch", "settee": "couch", "dining": "dining_table", "table": "dining_table",
        "tv": "tv", "television": "tv", "screen": "tv", "trash": "trash_can", "bin": "trash_can",
        "piano": "piano", "poker": "poker_table"}

    def resolve(self, text):
        """Best named place for a phrase ('the front door', 'by the sofa'): whole words only, never substrings."""
        words = __import__("re").findall(r"[a-z]+", text.lower().replace("_", " "))
        votes = {}
        for w in words:
            kind = self.SYNONYMS.get(w)
            if kind:
                votes[kind] = votes.get(kind, 0) + 1
        if not votes:
            return None
        kind = max(votes, key=votes.get)
        names = [k for k in self.place_names() if self.m.places[k].get("landmark", k) == kind or k == kind]
        return names[0] if names else None

    def where_am_i(self):
        if self.nav.T is None:
            return "Not localized yet."
        est = self.nav.estimate()
        near = sorted(((math.dist(est[:2], p["object_xy"]), k) for k, p in self.m.places.items()
                       if p.get("kind") == "landmark"))[:2]
        return "; ".join(f"{d:.1f} m from the {pretty(k)}" for d, k in near)

    def go_to(self, place):
        k = self.resolve(place)
        if k is None:
            return f"Unknown place '{place}'. Known places: {', '.join(self.place_names())}."
        if self.proposed != k:  # safety: never walk off on the first mention; the guest must agree
            self.proposed = k
            return (f"Not moving yet. Tell the guest the {pretty(k)} is {self.distance(k)} away and ask them to "
                    "confirm; call go_to again only after they agree.")
        self.pending, self.proposed = k, None
        self.state = f"about to walk to {k}"
        return f"OK: starting to walk to the {pretty(k)} now. Tell the guest to walk with you."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--script", nargs="*", help="scripted guest lines (typed), for recording")
    ap.add_argument("--spawn", default="random", help="random | <place> | x,y,yaw_deg")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--rerun-save", default=None)
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--robot", default="sim", choices=["sim", "real"])
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="real: robot_server.py on the K1's Jetson")
    ap.add_argument("--start-at", help="real: known start place (skip the look-around search), e.g. front_entrance")
    ap.add_argument("--yaw", type=float, default=None, help="real: heading at --start-at (deg, map frame)")
    ap.add_argument("--video", help="record a captioned, voiced chase-cam video of the session (mp4)")
    a = ap.parse_args()

    import conversation as conv_mod
    import navigator as nv
    import scan_map
    from landmarks import reachable

    os.makedirs(conv_mod.RUN_DIR, exist_ok=True)
    m = scan_map.Map(a.map_dir)
    reach = reachable(m)
    m.places.pop("start", None)
    field = nv.Field(m)
    rng = np.random.default_rng(a.seed)
    if a.spawn == "random":
        spawn = nv.sample_spawn(m, field, reach, rng)
    elif a.spawn in m.places:  # nearest spot with 0.5 m clearance (room to turn around)
        from scipy import ndimage

        clear = ndimage.distance_transform_edt(~(m.obstacle | ~m.free)) * 0.05
        rr, cc = np.nonzero(reach & (clear >= 0.5))
        px, py = m.places[a.spawn]["xy"]
        i = int(np.argmin(np.hypot(m.ox + (cc + 0.5) * 0.05 - px, m.oy + (rr + 0.5) * 0.05 - py)))
        spawn = (*m.xy(rr[i], cc[i]), float(rng.uniform(-math.pi, math.pi)))
    else:
        x, y, yaw = (float(v) for v in a.spawn.split(","))
        spawn = (x, y, math.radians(yaw))

    # Gemma first: LiteRT's GPU context must exist before MuJoCo's GL context (macOS)
    voice = Voice(enabled=not a.mute)
    nav = None
    talk = conv_mod.Conversation(None, None)
    if a.robot == "real":
        from real_robot import RealRobot

        robot = RealRobot(a.url)
    else:
        robot = nv.SimRobot(m, field, spawn, rng, gif=bool(a.video))
    if a.video and a.robot == "sim":
        voice.clock = lambda: robot.t
    dash = None
    if a.rerun or a.rerun_save:
        from dashboard import Dash

        dash = Dash(m, save=a.rerun_save)
    nav = nv.Navigator(m, field, reach, dash=dash)
    nav.truth = a.robot == "sim"
    tools = NavTools(nav, m)
    body = Body(robot, voice)
    talk.robot, talk.nav = body, tools
    talk.tools = conv_mod.make_tools(body, tools)
    com = Commentator(m, voice)
    log_ = nav.log
    nav.log = lambda s: log_(s)
    emit_ = nav.emit

    def emit(kind, **kw):
        emit_(kind, **kw)
        com.on_event(kind, kw, nav)

    nav.emit = emit

    try:
        if a.robot == "real" and a.start_at:
            nav.robot = robot
            yaw = a.yaw if a.yaw is not None else m.places[a.start_at].get("yaw_deg", 90.0)
            nav.T = nv.compose((*m.place(a.start_at), math.radians(yaw)), nv.inverse(robot.odom))
            nav.correct()
            nav.confidence = nv.UNSURE
        else:
            voice.say("Hello! Let me get my bearings first.", wait=False)
            nav.localize(robot)
        lines = list(a.script or [])
        if nav.confidence < nv.UNSURE:  # not sure: ask the guest instead of guessing
            voice.say("I'm not quite sure where we are. Which room are we in, or what's near us?", wait=True)
            idle(robot, "I'm not quite sure where we are. Which room are we in, or what's near us?")
            got = next_line(a, lines, conv_mod, voice, robot)
            heard = got
            if got.endswith(".wav") and os.path.exists(got):
                heard = talk.turn(audio_path=got)["heard"]
            k = tools.resolve(heard)
            if k:
                kind = m.places[k].get("landmark", k)
                nav.relocalize_near([p.get("object_xy", p["xy"]) for p in m.places.values()
                                     if p.get("landmark", None) == kind] or [m.places[k]["xy"]])
                voice.say(f"Thank you, that helps.", wait=False)
                idle(robot, "Thank you, that helps.")
        tools.state = "standing still, localized"
        hello = f"Okay, I know where I am: {tools.where_am_i().split(';')[0]}. Where would you like to go?"
        voice.say(hello, wait=True)
        idle(robot, hello)
        while True:
            if a.script is not None and not lines:
                break
            got = next_line(a, lines, conv_mod, voice, robot)
            if not got:
                continue
            if got.lower() in ("q", "quit", "exit"):
                break
            if got.lower() in ("s", "stop"):
                print("  " + body.stop())
                continue
            r = talk.turn(audio_path=got, text=None) if got.endswith(".wav") and os.path.exists(got) \
                else talk.turn(text=got)
            heard = (r.get("heard") or got).lower()
            if not tools.pending and tools.proposed and YES.search(heard):  # guest agreed to the proposal
                tools.pending, tools.proposed = tools.proposed, None
            elif not tools.pending and not tools.proposed:  # guest named a place; Gemma didn't call go_to
                k = tools.resolve(heard)
                if k:
                    tools.proposed = k
            if tools.pending and "?" in r["reply"]:  # the trip is starting: don't ask again
                r["reply"] = f"Wonderful, let's go to the {pretty(tools.pending)}."
            voice.say(r["reply"], urgent=True)
            idle(robot, r["reply"])
            if tools.pending:
                goal, tools.pending = tools.pending, None
                voice.wait()
                res = walk(nav, robot, goal, com)
                print(f"  trip: {res['result']} ({res.get('path_m')} m planned)", flush=True)
                tools.state = f"standing at the {pretty(goal)}"
                voice.wait()
    finally:
        voice.wait()
        if a.video and a.robot == "sim":
            robot.step(int(3.0 / robot.dt))
            write_video(a.video, robot, voice, m)
        robot.close()
        print(f"log: {os.path.relpath(conv_mod.RUN_DIR, HERE)}", flush=True)
        sys.stdout.flush()
        os._exit(0)


def next_line(a, lines, conv_mod, voice=None, robot=None):
    if a.script is not None:
        got = lines.pop(0) if lines else ""
        print(f"\nguest> {got}")
        if voice is not None and voice.clock and got:
            voice.lines.append((robot.t, "Guest", got))
            idle(robot, got)
        return got
    return input("\nyou> ").strip() if a.text else conv_mod.record_ptt()


def idle(robot, text):
    """Let the twin run (robot standing) for as long as `text` takes to say, so the video keeps time."""
    if robot.gif:
        robot.drive(0, 0)
        robot.step(int((0.8 + len(text.split()) / 2.6) / robot.dt))


def write_video(path, robot, voice, m, k1_voice="Ava (Premium)", guest_voice="Moira"):
    """Chase cam + mini-map + captions, with the spoken lines (macOS `say`) mixed in at their sim times."""
    import tempfile

    from PIL import Image, ImageDraw, ImageFont

    frames, meta = robot.frames, robot.frame_meta
    if not frames:
        return
    W, H = 1280, 720
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 30)
        small = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 20)
    except OSError:
        font = small = ImageFont.load_default()
    # mini-map
    h0, w0 = m.free.shape
    base = np.full((h0, w0, 3), 30, np.uint8)
    base[m.free] = (235, 235, 230)
    base[m.obstacle] = (70, 72, 85)
    mm = Image.fromarray(base[::-1])
    scale = 300 / max(w0, h0)
    mm = mm.resize((int(w0 * scale), int(h0 * scale)))
    tmp = tempfile.mkdtemp()
    fps = 4  # one frame per 0.25 s of sim time = real time
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", "fps=24", "-c:v", "libx264", "-crf", "18",
                           "-pix_fmt", "yuv420p", os.path.join(tmp, "v.mp4")], stdin=subprocess.PIPE)
    trail = []
    for img, (t, pose) in zip(frames, meta):
        im = Image.fromarray(img).resize((W, H))
        d = ImageDraw.Draw(im, "RGBA")
        # mini-map with trail
        trail.append(pose[:2])
        mmi = mm.copy()
        md = ImageDraw.Draw(mmi)
        pts = [((x - m.ox) / 0.05 * scale, (h0 - (y - m.oy) / 0.05) * scale) for x, y in trail]
        if len(pts) > 1:
            md.line(pts, fill=(42, 157, 143), width=3)
        px, py = pts[-1]
        md.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(230, 57, 70))
        im.paste(mmi, (W - mmi.width - 20, 20))
        d.rectangle([W - mmi.width - 20, 20, W - 20, 20 + mmi.height], outline=(255, 255, 255, 200), width=2)
        d.text((24, 20), "Booster K1 · AGI House digital twin · Gemma 4 on-device", font=small, fill=(255, 255, 255))
        d.text((24, 46), f"t = {t:5.1f} s (simulation)", font=small, fill=(220, 220, 220))
        # caption: the latest line said within the last 5 s
        cur = [(lt, who, txt) for lt, who, txt in voice.lines if lt <= t < lt + 5.0]
        if cur:
            _, who, txt = cur[-1]
            cap = f"{who}: {txt}"
            words, rows, row = cap.split(), [], ""
            for w in words:
                if d.textlength(row + " " + w, font=font) > W - 160:
                    rows.append(row)
                    row = w
                else:
                    row = (row + " " + w).strip()
            rows.append(row)
            y0 = H - 40 - 42 * len(rows)
            d.rectangle([60, y0 - 14, W - 60, H - 30], fill=(0, 0, 0, 160))
            for i, r in enumerate(rows):
                col = (255, 214, 102) if who == "K1" else (160, 220, 255)
                d.text((80, y0 + 42 * i), r, font=font, fill=col)
        ff.stdin.write(np.asarray(im.convert("RGB")).tobytes())
    ff.stdin.close()
    ff.wait()
    # voices
    inputs, filters = [], []
    for i, (t, who, txt) in enumerate(voice.lines):
        f = os.path.join(tmp, f"l{i}.aiff")
        subprocess.run(["say", "-v", k1_voice if who == "K1" else guest_voice, "-r", "165", "-o", f, txt],
                       check=False)
        if os.path.exists(f):
            inputs += ["-i", f]
            n = len(inputs) // 2
            filters.append(f"[{n}:a]adelay={int(t * 1000)}|{int(t * 1000)}[a{n}]")
    if inputs:
        n = len(inputs) // 2
        fc = ";".join(filters) + ";" + "".join(f"[a{k}]" for k in range(1, n + 1)) + \
            f"amix=inputs={n}:normalize=0[aout]"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(tmp, "v.mp4"), *inputs,
                        "-filter_complex", fc, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac",
                        "-shortest", path], check=False)
    else:
        shutil.copy(os.path.join(tmp, "v.mp4"), path)
    print(f"video -> {path} ({len(frames) / fps:.0f} s)", flush=True)


def walk(nav, robot, goal, com):
    """navigator.go_to with commentary ticks (wraps robot.step so ticks run inside the control loop)."""
    step_ = robot.step
    state = {"next": 0.0}

    def step(n=1):
        step_(n)
        if robot.t >= state["next"] and getattr(nav, "_path", None):
            state["next"] = robot.t + 0.5
            com.tick(nav, goal, nav._path, nav._wp)

    robot.step = step
    try:
        res = nav.go_to(goal)
    finally:
        robot.step = step_
    if res["ok"]:
        (robot.sim if hasattr(robot, "sim") else robot).wave()
        robot.step(int(2.0 / robot.dt))
    return res


if __name__ == "__main__":
    main()
