"""Gemma-on-device robot brain: camera frame -> Gemma 4 (Ollama, localhost) -> tool call -> robot.

    # sim robot, sim head camera (default)
    .venv/bin/python brain.py --goal "find the red box, walk up to it, then wave"
    # sim robot, Mac webcam as the eyes
    .venv/bin/python brain.py --camera webcam --goal "if you see a person, wave and say hi"
    # real K1 straight from the Mac over DDS (community booster-sdk), webcam as eyes
    .venv/bin/python brain.py --robot booster_sdk --camera webcam --goal "..."
    # real K1 through robot_server.py running on the robot / a Linux box
    .venv/bin/python brain.py --robot remote --url http://192.168.x.x:8765 --goal "..."

Type a new goal + Enter at any time to change it; type `q` or Ctrl-C to stop.
All inference is local (Ollama on 127.0.0.1). Safety clamp + 1 Hz rate limit live in robot.SafeRobot.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import ollama
from PIL import Image

from robot import make_robot
from tools import TOOLS

SYSTEM = """You are the brain of a Booster K1, a 95 cm humanoid robot. The image is what your head camera sees right now.
Pursue the GOAL by calling exactly ONE tool per turn. Never answer in plain text.
How to move:
- Target visible near the image centre and still small/far: walk(vx=0.4, seconds=2).
- Target visible but left of centre: turn(degrees=15..30). Right of centre: turn(degrees=-15..-30).
- Target fills the bottom of the image (very close): you have arrived; do the final part of the goal (e.g. wave, say).
- Target not visible: turn(degrees=45) to scan.
- Don't repeat the same look() twice in a row; looking doesn't move you."""


def to_jpeg(rgb: np.ndarray, max_side: int = 512) -> bytes:
    im = Image.fromarray(rgb)
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=85)
    return b.getvalue()


class Webcam:
    def __init__(self, index=0):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("webcam not available (macOS: allow camera access for your terminal app)")

    def get_frame(self):
        for _ in range(3):  # drop stale buffered frames
            self.cap.grab()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return self.cv2.cvtColor(f, self.cv2.COLOR_BGR2RGB)


def stdin_goals(q: queue.Queue):
    for line in sys.stdin:
        q.put(line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e2b-it-qat", help="or gemma4:e4b-it-qat (better tool use, ~1.7x slower)")
    ap.add_argument("--robot", default="sim", choices=["sim", "remote", "booster_sdk", "boosteros"])
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="robot_server.py address for --robot remote")
    ap.add_argument("--frame-url", default=None, help="JPEG endpoint for --robot booster_sdk (else use --camera webcam)")
    ap.add_argument("--camera", default="robot", choices=["robot", "webcam"], help="robot = sim head cam / real K1 cam")
    ap.add_argument("--goal", default="Find the red box, walk up close to it, then wave.")
    ap.add_argument("--steps", type=int, default=0, help="stop after N decisions (0 = run forever)")
    ap.add_argument("--period", type=float, default=1.0, help="min seconds between decisions (~1 Hz)")
    ap.add_argument("--viewer", action="store_true", help="MuJoCo window (run with .venv/bin/mjpython)")
    ap.add_argument("--mute", action="store_true", help="don't speak say() out loud")
    ap.add_argument("--think", action="store_true", help="enable Gemma thinking (slower, sometimes smarter)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()

    if "127.0.0.1" not in a.host and "localhost" not in a.host:
        print("WARNING: --host is not localhost; make sure it's your own on-device server (no cloud).")

    kw = {}
    if a.robot == "sim":
        kw = {"viewer": a.viewer, "speak": not a.mute}
    elif a.robot == "remote":
        kw = {"url": a.url}
    elif a.robot == "booster_sdk":
        kw = {"frame_url": a.frame_url, "speak": not a.mute}
    elif a.robot == "boosteros":
        kw = {"speak": not a.mute}
    robot = make_robot(a.robot, min_interval=a.period, **kw)
    cam = Webcam() if a.camera == "webcam" else robot

    client = ollama.Client(host=a.host)
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "log.jsonl"), "w")
    print(f"model={a.model} robot={a.robot} camera={a.camera} -> logging to {run_dir}")

    goals: queue.Queue = queue.Queue()
    threading.Thread(target=stdin_goals, args=(goals,), daemon=True).start()
    goal, history, step = a.goal, [], 0
    print(f"GOAL: {goal}")
    try:
        while True:
            t_loop = time.monotonic()
            while not goals.empty():
                g = goals.get()
                if g.lower() in ("q", "quit", "exit"):
                    raise KeyboardInterrupt
                if g:
                    goal, history = g, []
                    robot.stop()
                    print(f"NEW GOAL: {goal}")
            frame = cam.get_frame()
            if frame is None:
                print("no frame; stopping"); robot.stop(); time.sleep(0.5); continue
            jpg = to_jpeg(frame)
            status = robot.status()
            recent = "; ".join(history[-6:]) or "none"
            if len(history) >= 3 and len(set(h.split(" ")[0] for h in history[-3:])) == 1:
                recent += " (you are repeating yourself - choose a different action that makes progress)"
            user = f"GOAL: {goal}\nRobot status: {json.dumps(status)}\nYour recent actions: {recent}\nWhat is your next single action?"
            t0 = time.perf_counter()
            resp = client.chat(model=a.model, think=a.think, tools=TOOLS, keep_alive="30m",
                               options={"temperature": 0.2, "num_ctx": 4096},
                               messages=[{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": user, "images": [jpg]}])
            dt = time.perf_counter() - t0
            calls = resp.message.tool_calls or []
            if calls:
                c = calls[0]  # one action per tick
                result = robot.call(c.function.name, dict(c.function.arguments))
                history.append(result)
                act = {"name": c.function.name, "arguments": dict(c.function.arguments)}
            else:
                result = robot.stop()  # plain text = no valid action -> stand still
                act = {"text": (resp.message.content or "")[:200]}
            tps = resp.eval_count / (resp.eval_duration / 1e9) if resp.eval_duration else 0
            print(f"[{step:03d}] gemma {dt:.2f}s ({tps:.0f} tok/s) -> {act} | {status}")
            Image.fromarray(frame).save(os.path.join(run_dir, f"{step:03d}_eye.jpg"))
            if a.robot == "sim" and step % 2 == 0:
                Image.fromarray(robot.b.get_chase_frame()).save(os.path.join(run_dir, f"{step:03d}_chase.jpg"))
            log.write(json.dumps({"step": step, "goal": goal, "latency_s": round(dt, 3), "action": act,
                                  "result": result, "status": status}) + "\n"); log.flush()
            if status.get("fallen"):
                print("robot fell over; stopping"); break
            step += 1
            if a.steps and step >= a.steps:
                break
            time.sleep(max(0.0, a.period - (time.monotonic() - t_loop)))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        robot.close()
        log.close()
        print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
