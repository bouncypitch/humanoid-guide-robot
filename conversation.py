"""Talk to the robot: your voice + its head-camera view -> Gemma 4 (LiteRT-LM, on-device) -> spoken reply / actions.

    .venv/bin/python conversation.py                 # sim K1, push-to-talk with Enter
    .venv/bin/python conversation.py --text          # type instead of speaking (loud venue)
    .venv/bin/python conversation.py --wav q.wav     # one turn from a recorded question (tests)
    .venv/bin/python conversation.py --robot remote --url http://<jetson>:8765

Push-to-talk: press Enter to start talking, Enter again to stop. Type `s` + Enter to stop
the robot immediately (no model involved), `q` to quit.

Gemma hears the audio directly (no separate speech recogniser) and sees the current head-camera
frame each turn. It answers in 1-2 spoken sentences or calls a tool (look, wave, turn, go_to,
where_am_i, stop). Tools run automatically and Gemma then says what it did.
Each turn is logged to runs/<ts>/conversation.jsonl for the dashboard and the story.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models/gemma-4-E2B-it.litertlm")
RATE = 16000

PERSONA = """You are the voice and brain of a Booster K1, a friendly 95 cm humanoid robot working as a guide in a house.
You help guests, especially elderly guests, find their way to places like the kitchen or the restroom.
You see through your head camera (the image) and hear the guest (the audio).

How to reply:
- First line exactly: heard: <the guest's words, transcribed>
- Then your spoken reply: 1-2 short, warm, plain sentences. No lists, no emoji, no markdown.
- Describe only what is really in the image. If unsure, say so.
- Before walking somewhere, ask the guest to confirm ("Shall we go?"). Only call go_to after they agree.
- If a request is ambiguous, ask one short clarifying question.
- Use tools only when the guest asks you to do something physical or asks where you are."""

STATE_TMPL = "[robot state] {state}\n[known places] {places}\n[recent conversation]\n{history}"


# ---------------------------------------------------------------- audio
def record_ptt() -> str | None:
    """Press Enter to start, Enter to stop. Returns a 16 kHz mono wav path, or a command string."""
    import sounddevice as sd

    cmd = input("\n[Enter] talk  |  type text and Enter  |  s = STOP robot  |  q = quit > ").strip()
    if cmd:
        return cmd
    chunks = []

    def cb(indata, frames, t, status):  # noqa: ARG001
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=cb):
        input("  listening... press Enter when done ")
    audio = np.concatenate(chunks) if chunks else np.zeros((1, 1), np.int16)
    if len(audio) < RATE * 0.3:
        print("  (too short)")
        return ""
    path = tempfile.mktemp(suffix=".wav", dir=RUN_DIR)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(audio.tobytes())
    return path


def speak(robot, text: str):
    """Blocking speech so we never listen while talking. Sim/Mac: `say`; real robot: its speaker."""
    if not text:
        return
    print(f"\n  K1: {text}")
    if robot.b.name == "sim" and platform.system() == "Darwin" and shutil.which("say"):
        subprocess.run(["say", "-r", "170", text], check=False)
    else:
        robot.say(text)


# ---------------------------------------------------------------- tools (Gemma calls these)
def make_tools(robot, nav):
    def look(yaw_deg: float, pitch_deg: float) -> str:
        """Turn the head to look somewhere. yaw_deg positive looks left (-55..55); pitch_deg positive looks down (-15..45)."""
        return robot.look(yaw_deg, pitch_deg)

    def wave() -> str:
        """Wave hello with the right hand."""
        return robot.wave()

    def turn(degrees: float) -> str:
        """Turn the whole body in place. Positive degrees turns left (-90..90)."""
        return robot.turn(degrees)

    def stop() -> str:
        """Stop all motion immediately."""
        return robot.stop()

    def where_am_i() -> str:
        """Report where the robot is in the house."""
        return nav.where_am_i()

    def go_to(place: str) -> str:
        """Walk to a named place in the house, guiding the guest. Only call after the guest agreed."""
        return nav.go_to(place)

    return [look, wave, turn, stop, where_am_i, go_to]


class NavStub:
    """Placeholder until navigator.py lands: knows the map's place names, reports honestly."""

    def __init__(self, map_dir: str | None):
        self.places = {}
        if map_dir and os.path.exists(os.path.join(map_dir, "places.json")):
            self.places = json.load(open(os.path.join(map_dir, "places.json")))
        self.state = "standing still"

    def place_names(self):
        # Gemma's walkthrough landmarks are the destinations; "_2" duplicates and no-go zones are not offered
        return [p for p, v in self.places.items() if p != "start" and v.get("kind") != "nogo"
                and not p.rsplit("_", 1)[-1].isdigit()]

    def where_am_i(self) -> str:
        return "I haven't localized yet; my navigation module is still starting up."

    def go_to(self, place: str) -> str:
        if place not in self.places:
            return f"Unknown place '{place}'. Known places: {', '.join(self.place_names()) or 'none'}."
        self.state = f"heading to {place}"
        return f"Navigation to {place} is not connected yet in this build; tell the guest honestly."


# ---------------------------------------------------------------- the loop
class Conversation:
    def __init__(self, robot, nav, model=MODEL, backend="gpu"):
        import litert_lm as lm

        self.lm, self.robot, self.nav = lm, robot, nav
        t = time.perf_counter()
        self.engine = lm.Engine(model, backend=lm.Backend.GPU() if backend == "gpu" else lm.Backend.CPU(),
                                vision_backend=lm.Backend.CPU(), audio_backend=lm.Backend.CPU(),
                                enable_benchmark=True)
        print(f"Gemma 4 E2B (LiteRT-LM, {backend}) loaded in {time.perf_counter() - t:.1f}s")
        self.tools = make_tools(robot, nav)
        self.history: list[tuple[str, str]] = []
        self.log = open(os.path.join(RUN_DIR, "conversation.jsonl"), "a")

    def turn(self, audio_path: str | None = None, text: str | None = None) -> dict:
        frame = self.robot.get_frame()
        img_path = os.path.join(RUN_DIR, f"frame_{len(self.history):03d}.jpg")
        Image.fromarray(frame).save(img_path, quality=85)
        hist = "\n".join(f"guest: {u}\nrobot: {r}" for u, r in self.history[-8:]) or "(none yet)"
        state = STATE_TMPL.format(state=f"{self.nav.state}; {self.robot.status()}",
                                  places=", ".join(self.nav.place_names()) or "unknown", history=hist)
        content = [{"type": "image", "path": img_path}]
        if audio_path:
            content.append({"type": "audio", "path": os.path.abspath(audio_path)})
            content.append({"type": "text", "text": state + "\nThe guest just spoke (audio). Reply to them."})
        else:
            content.append({"type": "text", "text": state + f"\nThe guest says: {text}"})
        conv = self.engine.create_conversation(system_message=PERSONA, tools=self.tools,
                                               automatic_tool_calling=True)
        t0 = time.perf_counter()
        msg = conv.send_message({"role": "user", "content": content})
        dt = time.perf_counter() - t0
        try:
            bi = conv.get_benchmark_info()
            ttft = round(bi.time_to_first_token_in_second, 2)
        except Exception:  # noqa: BLE001
            ttft = None
        conv.close()
        raw = _text_of(msg)
        heard, reply = _split(raw, text)
        self.history.append((heard, reply))
        rec = {"t": time.time(), "heard": heard, "reply": reply, "latency_s": round(dt, 2),
               "ttft_s": ttft, "frame": os.path.basename(img_path), "robot_log": self.robot.log[-3:]}
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        print(f"  (heard: {heard!r}; {dt:.1f}s, first token {ttft}s)")
        return rec


def _text_of(msg) -> str:
    c = msg.contents if hasattr(msg, "contents") else msg
    if isinstance(c, str):
        return c
    try:
        parts = [getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None) for p in c]
        return "\n".join(p for p in parts if p)
    except TypeError:
        return str(c)


def _split(raw: str, typed: str | None):
    heard, lines = typed or "", []
    for ln in raw.strip().splitlines():
        m = re.match(r"\s*heard\s*:\s*(.*)", ln, re.I)
        if m and not heard:
            heard = m.group(1).strip()
        elif m:
            continue
        elif ln.strip():
            lines.append(ln.strip())
    reply = re.sub(r"[*_#`]", "", " ".join(lines)).strip()
    return heard, reply


RUN_DIR = os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S") + "-talk")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="sim", choices=["sim", "remote", "booster_sdk", "boosteros"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--map", default=os.path.join(HERE, "maps/agihouse"))
    ap.add_argument("--text", action="store_true", help="type instead of push-to-talk")
    ap.add_argument("--wav", default=None, help="run a single turn from this wav and exit")
    ap.add_argument("--ask", default=None, help="run a single typed turn and exit")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)

    from robot import make_robot

    # Load Gemma before the sim: creating LiteRT's GPU (WebGPU/Metal) engine after MuJoCo's
    # GL context exists segfaults on macOS.
    talk = Conversation(None, NavStub(a.map), backend=a.backend)
    kw = {"speak": False} if a.robot == "sim" else ({"url": a.url} if a.robot == "remote" else {})
    robot = make_robot(a.robot, **kw)
    talk.robot = robot
    talk.tools = make_tools(robot, talk.nav)
    try:
        if a.wav or a.ask:
            r = talk.turn(audio_path=a.wav, text=a.ask)
            speak(robot, r["reply"])
            return
        speak(robot, "Hello! I'm your guide. Where would you like to go?")
        while True:
            got = input("\nyou> ").strip() if a.text else record_ptt()
            if got is None or got == "":
                continue
            if got.lower() in ("q", "quit", "exit"):
                break
            if got.lower() in ("s", "stop"):  # hard stop: bypasses the model entirely
                print("  " + robot.stop())
                continue
            r = talk.turn(audio_path=got, text=None) if got.endswith(".wav") and os.path.exists(got) \
                else talk.turn(text=got)
            speak(robot, r["reply"])
    finally:
        robot.close()
        print(f"log: {os.path.relpath(RUN_DIR, HERE)}/conversation.jsonl", flush=True)
        # MuJoCo's renderer __del__ segfaults at interpreter shutdown when LiteRT's Metal
        # context is alive; everything is flushed, so skip Python teardown.
        os._exit(0)


if __name__ == "__main__":
    main()
