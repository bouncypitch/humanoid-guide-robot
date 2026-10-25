"""LiteRT-LM smoke test (for the "Best Use of LiteRT" prize): Gemma 4 E2B .litertlm
on the Mac, image (+ optional audio) + robot tools -> tool call. Fully on-device.

    .venv-litert/bin/python litert_bench.py                        # image + text
    .venv-litert/bin/python litert_bench.py --audio runs/cmd.wav   # image + spoken command
"""
import argparse
import os
import subprocess
import time

import litert_lm as lm

HERE = os.path.dirname(os.path.abspath(__file__))


# Tool signatures mirror tools.py; docstrings become the schema descriptions.
def walk(vx: float, vy: float = 0.0, seconds: float = 1.5) -> str:
    """Walk the robot. vx forward m/s (max 0.5), vy left m/s, for `seconds`.

    Args:
        vx: forward speed in m/s, -0.3..0.5
        vy: sideways speed in m/s, left positive
        seconds: duration 0.5..3
    """
    return "ok"


def turn(degrees: float) -> str:
    """Turn in place; positive = left.

    Args:
        degrees: -90..90
    """
    return "ok"


def look(yaw_deg: float, pitch_deg: float) -> str:
    """Point the head. yaw positive = left, pitch positive = down.

    Args:
        yaw_deg: -55..55
        pitch_deg: -15..45
    """
    return "ok"


def wave() -> str:
    """Wave hello with the right hand."""
    return "ok"


def say(text: str) -> str:
    """Speak a short sentence out loud.

    Args:
        text: under 20 words
    """
    return "ok"


def stop() -> str:
    """Stop all motion immediately."""
    return "ok"


SYSTEM = ("You are the brain of a small humanoid robot seeing through its head camera. "
          "Always act by calling exactly one tool.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models/gemma-4-E2B-it.litertlm"))
    ap.add_argument("--image", default=os.path.join(HERE, "runs/smoke_head.png"))
    ap.add_argument("--audio", default=None, help="wav file with a spoken command")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--prompt", default="Walk toward the red box.")
    a = ap.parse_args()

    t = time.perf_counter()
    be = lm.Backend.GPU() if a.backend == "gpu" else lm.Backend.CPU()
    engine = lm.Engine(a.model, backend=be, vision_backend=lm.Backend.CPU(),
                       audio_backend=lm.Backend.CPU() if a.audio else None,
                       enable_benchmark=True)  # compiled caches land next to the model
    print(f"engine load {time.perf_counter() - t:.1f}s ({a.backend})")

    prompts = [a.prompt, "Someone said hello. Greet them.", "Danger ahead!"]
    for p in prompts:
        conv = engine.create_conversation(system_message=SYSTEM,
                                          tools=[walk, turn, look, wave, say, stop],
                                          automatic_tool_calling=False)
        content = [{"type": "image", "path": os.path.abspath(a.image)}]
        if a.audio:
            content.append({"type": "audio", "path": os.path.abspath(a.audio)})
            p = "Follow the spoken command in the audio."
        content.append({"type": "text", "text": p})
        t = time.perf_counter()
        msg = conv.send_message({"role": "user", "content": content})
        dt = time.perf_counter() - t
        calls = [{"name": tc.name, "arguments": tc.arguments} for tc in (msg.tool_calls or [])]
        text = str(msg.contents) if msg.contents else ""
        out = {"tool_calls": calls, "text": text[:200], "channels": dict(msg.channels or {})}
        try:
            bi = conv.get_benchmark_info()
            bstr = (f"TTFT {bi.time_to_first_token_in_second:.2f}s | "
                    f"prefill {bi.last_prefill_tokens_per_second:.0f} tok/s | "
                    f"decode {bi.last_decode_tokens_per_second:.1f} tok/s")
        except Exception as e:  # noqa: BLE001
            bstr = f"(no benchmark info: {e})"
        print(f"\nPROMPT: {p}\n  {dt:.2f}s | {bstr}\n  -> {out}")
        conv.close()
        if a.audio:
            break


if __name__ == "__main__":
    main()
