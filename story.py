"""Narratable build log for the demo: numbered checkpoints with artifacts + a code snapshot.

    python story.py checkpoint "K1 walks in MuJoCo" --note "Booster's RL policy..." \
        --attach runs/smoke_head.png --metric "walk speed=0.5 m/s" --say "It already walks."
    python story.py index      # regenerate story/index.md
    python story.py html       # story/story.html: self-contained, one section per checkpoint
    python story.py list

Each checkpoint is story/NN-slug/ with README.md, meta.json, copied artifacts and
snapshot.tar.gz (code + docs only: no models, venvs, vendor repos or captures).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import mimetypes
import os
import re
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
STORY = os.path.join(HERE, "story")
SNAP_EXT = {".py", ".md", ".swift", ".yml", ".yaml", ".kt", ".kts", ".json", ".plist", ".toml", ".txt"}
SNAP_SKIP = {".venv", ".venv-litert", "models", "booster_assets", "booster_deploy",
             "booster_robotics_sdk", "story", "captures", "runs", "logs", "__pycache__",
             "build", "DerivedData", ".git", "reference", "scans"}
INLINE_MAX = 6 * 1024 * 1024  # don't inline artifacts bigger than this into story.html


def _checkpoints():
    if not os.path.isdir(STORY):
        return []
    out = []
    for d in sorted(os.listdir(STORY)):
        m = os.path.join(STORY, d, "meta.json")
        if re.match(r"^\d{2}-", d) and os.path.exists(m):
            meta = json.load(open(m))
            meta["dir"] = d
            out.append(meta)
    return out


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def _snapshot(dest):
    with tarfile.open(dest, "w:gz") as tar:
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if d not in SNAP_SKIP and not d.endswith(".xcodeproj")]
            for f in files:
                p = os.path.join(root, f)
                if os.path.splitext(f)[1] in SNAP_EXT and os.path.getsize(p) < 2_000_000:
                    tar.add(p, arcname=os.path.relpath(p, HERE))


def checkpoint(title, note="", attach=(), metrics=(), say="", when=None):
    os.makedirs(STORY, exist_ok=True)
    n = len(_checkpoints()) + 1
    d = os.path.join(STORY, f"{n:02d}-{_slug(title)}")
    os.makedirs(d)
    files = []
    for a in attach:
        if not os.path.exists(a):
            print(f"  ! missing attachment {a}", file=sys.stderr)
            continue
        name = os.path.basename(a)
        if name in files:  # same basename from different dirs
            name = f"{len(files)}_{name}"
        shutil.copy2(a, os.path.join(d, name))
        files.append(name)
    met = {}
    for m in metrics:
        k, _, v = m.partition("=")
        met[k.strip()] = v.strip()
    meta = {"n": n, "title": title, "note": note, "say": say, "metrics": met, "files": files,
            "time": when or dt.datetime.now().isoformat(timespec="minutes")}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    _snapshot(os.path.join(d, "snapshot.tar.gz"))
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write(f"# {n:02d}. {title}\n\n_{meta['time']}_\n\n{note}\n\n")
        if met:
            f.write("| Metric | Value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in met.items()) + "\n")
        if say:
            f.write(f"**Say on stage:** {say}\n\n")
        for fn in files:
            f.write(f"![{fn}]({fn})\n\n" if fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif")) else f"- [{fn}]({fn})\n")
    index()
    print(f"checkpoint {n:02d}: {title} -> {os.path.relpath(d, HERE)}")
    return d


def index():
    cps = _checkpoints()
    lines = ["# Wayfinder: build story\n", "Drop it anywhere in a home. It finds its way, and talks you through it.\n"]
    for c in cps:
        met = ", ".join(f"{k}: {v}" for k, v in c["metrics"].items())
        img = next((f for f in c["files"] if f.lower().endswith((".png", ".jpg", ".gif"))), None)
        lines.append(f"## {c['n']:02d}. [{c['title']}]({c['dir']}/README.md)\n")
        lines.append(f"_{c['time']}_ {('| ' + met) if met else ''}\n")
        if c["note"]:
            lines.append(c["note"] + "\n")
        if img:
            lines.append(f"<img src=\"{c['dir']}/{img}\" width=\"420\">\n")
    os.makedirs(STORY, exist_ok=True)
    open(os.path.join(STORY, "index.md"), "w").write("\n".join(lines))


def _data_uri(path):
    if os.path.getsize(path) > INLINE_MAX:
        return None
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def render_html():
    cps = _checkpoints()
    secs = []
    for c in cps:
        imgs = []
        for fn in c["files"]:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                uri = _data_uri(os.path.join(STORY, c["dir"], fn))
                if uri:
                    imgs.append(f'<figure><img src="{uri}" alt="{html.escape(fn)}"><figcaption>{html.escape(fn)}</figcaption></figure>')
        mets = "".join(f'<div class="m"><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>'
                       for k, v in c["metrics"].items())
        say = f'<p class="say">“{html.escape(c["say"])}”</p>' if c.get("say") else ""
        secs.append(f'''<section><div class="num">{c["n"]:02d}</div><h2>{html.escape(c["title"])}</h2>
<p class="time">{html.escape(c["time"])}</p><p>{html.escape(c["note"])}</p>
<div class="mets">{mets}</div>{say}<div class="imgs">{"".join(imgs)}</div></section>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Wayfinder Build Story</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--acc:#2a6fdb;--card:#fff;--line:#e6e3dc}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#141413;--fg:#ecebe6;--mut:#a3a29c;--acc:#7aa7ff;--card:#1e1e1c;--line:#33322e}}}}
:root[data-theme="dark"]{{--bg:#141413;--fg:#ecebe6;--mut:#a3a29c;--acc:#7aa7ff;--card:#1e1e1c;--line:#33322e}}
body{{margin:0;background:var(--bg);color:var(--fg);font:17px/1.55 -apple-system,system-ui,sans-serif}}
header,section{{max-width:980px;margin:0 auto;padding:40px 16px;border-bottom:1px solid var(--line)}}
h1{{font-size:clamp(28px,5vw,44px);margin:0 0 8px}} h2{{margin:4px 0;font-size:clamp(22px,3.5vw,30px)}}
.num{{color:var(--acc);font-weight:700;letter-spacing:.08em}} .time{{color:var(--mut);margin:0 0 12px;font-size:14px}}
.mets{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}}
.m{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;display:flex;flex-direction:column}}
.m b{{font-size:20px}} .m span{{color:var(--mut);font-size:13px}}
.say{{border-left:3px solid var(--acc);padding-left:12px;font-style:italic}}
.imgs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
figure{{margin:0}} img{{width:100%;border-radius:10px;border:1px solid var(--line);background:#fff}}
figcaption{{color:var(--mut);font-size:12px}}
</style></head><body><header><h1>Wayfinder</h1><p>Drop it anywhere in a home. It finds its way, and talks you through it.</p>
<p>Gemma 4, 100% on-device · Booster K1 · Google Gemma Hardware Hackathon, AGI House, Oct 24 2026</p></header>
{"".join(secs)}</body></html>'''
    out = os.path.join(STORY, "story.html")
    open(out, "w").write(page)
    print(f"{len(cps)} checkpoints -> {os.path.relpath(out, HERE)} ({os.path.getsize(out) // 1024} KB)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("checkpoint")
    p.add_argument("title"); p.add_argument("--note", default=""); p.add_argument("--say", default="")
    p.add_argument("--attach", nargs="*", default=[]); p.add_argument("--metric", action="append", default=[])
    p.add_argument("--time", default=None, help="override timestamp (ISO), for back-filled checkpoints")
    sub.add_parser("index"); sub.add_parser("html"); sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "checkpoint":
        checkpoint(a.title, a.note, a.attach, a.metric, a.say, a.time)
    elif a.cmd == "index":
        index()
    elif a.cmd == "html":
        render_html()
    else:
        for c in _checkpoints():
            print(f"{c['n']:02d}  {c['time']}  {c['title']}")


if __name__ == "__main__":
    main()
