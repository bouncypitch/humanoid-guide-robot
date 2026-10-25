"""Verify local Gemma 4 via Ollama: image + prompt + tools -> well-formed tool call.

Measures time-to-first-token (wall clock, streaming), decode tokens/sec and
prompt-eval speed (from Ollama's own counters). Everything runs on localhost.

    .venv/bin/python bench_gemma.py --model gemma4:e2b-it-qat --image runs/smoke_head.png
"""
import argparse
import json
import time

import ollama

from tools import TOOLS, TOOL_NAMES

SYSTEM = ("You are the brain of a small humanoid robot. You see through its head camera. "
          "Always act by calling exactly one tool. Never answer in plain text.")

PROMPTS = [
    "Walk toward the red box in front of you.",
    "Someone just said hello to you. Greet them.",
    "Look to your left to check for obstacles.",
    "Danger! Something is in your way.",
]


def fresh_image(path):
    """Return PNG bytes with a random pixel tweak so Ollama can't reuse the
    cached image prefix (a real camera loop sends a new frame every time)."""
    import io, random
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.putpixel((random.randrange(im.width), random.randrange(im.height)),
                tuple(random.randrange(256) for _ in range(3)))
    b = io.BytesIO(); im.save(b, format="PNG")
    return b.getvalue()


def run_once(client, model, image, prompt, num_ctx, think=False):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt, "images": [fresh_image(image)]}]
    t0 = time.perf_counter()
    ttft = None
    calls, text, final = [], "", None
    for chunk in client.chat(model=model, messages=msgs, tools=TOOLS, stream=True, think=think,
                             options={"temperature": 0, "num_ctx": num_ctx},
                             keep_alive="30m"):
        m = chunk.message
        if ttft is None and (m.content or m.tool_calls or m.thinking):
            ttft = time.perf_counter() - t0
        if m.content:
            text += m.content
        if m.tool_calls:
            calls += m.tool_calls
        if chunk.done:
            final = chunk
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    ok = bool(calls) and all(c.function.name in TOOL_NAMES and isinstance(c.function.arguments, dict)
                             for c in calls)
    return {
        "prompt": prompt,
        "tool_calls": [{"name": c.function.name, "arguments": dict(c.function.arguments)} for c in calls],
        "text": text.strip()[:200],
        "well_formed": ok,
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "prompt_tokens": final.prompt_eval_count,
        "prompt_tok_per_s": round(final.prompt_eval_count / (final.prompt_eval_duration / 1e9), 1)
        if final.prompt_eval_duration else None,
        "gen_tokens": final.eval_count,
        "gen_tok_per_s": round(final.eval_count / (final.eval_duration / 1e9), 1)
        if final.eval_duration else None,
        "load_s": round((final.load_duration or 0) / 1e9, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e2b-it-qat")
    ap.add_argument("--image", default="runs/smoke_head.png")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--think", action="store_true", help="enable Gemma 4 thinking (slower)")
    a = ap.parse_args()
    client = ollama.Client(host=a.host)
    print(f"model={a.model} image={a.image} think={a.think}")
    print("warm-up (loads weights) ...")
    w = run_once(client, a.model, a.image, PROMPTS[0], a.num_ctx, a.think)
    print(f"  cold: total {w['total_s']}s (load {w['load_s']}s)")
    results = [run_once(client, a.model, a.image, p, a.num_ctx, a.think) for p in PROMPTS]
    for r in results:
        print(json.dumps(r))
    n = len(results)
    print(f"SUMMARY {a.model}: well_formed {sum(r['well_formed'] for r in results)}/{n} | "
          f"mean TTFT {sum(r['ttft_s'] for r in results)/n:.2f}s | "
          f"mean total {sum(r['total_s'] for r in results)/n:.2f}s | "
          f"decode {sum(r['gen_tok_per_s'] or 0 for r in results)/n:.1f} tok/s | "
          f"prefill {sum(r['prompt_tok_per_s'] or 0 for r in results)/n:.0f} tok/s")


if __name__ == "__main__":
    main()
