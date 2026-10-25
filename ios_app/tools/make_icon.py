"""Draw the RGBD Capture app icon (1024x1024) into the asset catalog.

    ../.venv/bin/python tools/make_icon.py

Vivid indigo -> violet -> pink gradient with a soft glow, white camera-viewfinder corners,
and an isometric cube whose faces are LiDAR-style depth dots (cyan near, amber far).
iOS applies the rounded-corner mask itself, so the image is a full square.
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 2048  # supersample, downscaled to 1024 at the end
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "RGBDCapture", "Assets.xcassets", "AppIcon.appiconset")


def lerp(a, b, t):
    return a + (b - a) * t


def gradient():
    y, x = np.mgrid[0:S, 0:S] / (S - 1)
    t = np.clip((x * 0.55 + y * 0.75) / 1.3, 0, 1)[..., None]
    c0, c1, c2 = np.array([70, 92, 255]), np.array([167, 84, 247]), np.array([255, 70, 120])
    img = np.where(t < 0.5, lerp(c0, c1, t / 0.5), lerp(c1, c2, (t - 0.5) / 0.5))
    # soft cyan glow top-left and warm glow bottom-right
    for (cx, cy, r, col, a) in [(0.18, 0.12, 0.55, (90, 220, 255), 0.55), (0.9, 0.95, 0.5, (255, 190, 90), 0.35)]:
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r
        w = (np.clip(1 - d, 0, 1) ** 2 * a)[..., None]
        img = img * (1 - w) + np.array(col) * w
    return Image.fromarray(img.astype(np.uint8), "RGB")


def depth_color(t):
    """t=0 near (cyan) -> 1 far (amber)."""
    stops = [(0.0, (120, 245, 255)), (0.5, (255, 255, 255)), (1.0, (255, 214, 102))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return tuple(int(lerp(a, b, u)) for a, b in zip(c0, c1))
    return stops[-1][1]


def main():
    img = gradient().convert("RGBA")
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # viewfinder corners
    m, L, W = S * 0.16, S * 0.13, int(S * 0.028)
    for (x, y, sx, sy) in [(m, m, 1, 1), (S - m, m, -1, 1), (m, S - m, 1, -1), (S - m, S - m, -1, -1)]:
        d.line([(x, y + sy * L), (x, y), (x + sx * L, y)], fill=(255, 255, 255, 245), width=W, joint="curve")
        for (px, py) in [(x, y + sy * L), (x + sx * L, y), (x, y)]:
            d.ellipse([px - W / 2, py - W / 2, px + W / 2, py + W / 2], fill=(255, 255, 255, 245))

    # isometric cube of depth dots
    cx, cy, a = S * 0.5, S * 0.53, S * 0.205
    iso = lambda u, v, w: (cx + (u - v) * a * math.cos(math.pi / 6),
                           cy + (u + v) * a * 0.5 - w * a)
    n = 6
    faces = [  # (point fn, depth fn) for top, left, right faces
        (lambda i, j: iso(i, j, 1.0), lambda i, j: 0.5 * (i + j)),
        (lambda i, j: iso(i, 1.0, 1 - j), lambda i, j: 0.3 + 0.7 * i),
        (lambda i, j: iso(1.0, i, 1 - j), lambda i, j: 0.3 + 0.7 * (1 - i)),
    ]
    shade = [1.0, 0.82, 0.92]
    for (pt, dep), s in zip(faces, shade):
        for a_ in range(n):
            for b_ in range(n):
                i, j = a_ / (n - 1), b_ / (n - 1)
                x, y = pt(i, j)
                col = tuple(int(c * s) for c in depth_color(dep(i, j)))
                r = S * 0.0125 * (1.15 - 0.3 * dep(i, j))
                d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))
    # crisp cube edges
    E = [((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (1, 1, 1)), ((1, 1, 1), (0, 1, 1)), ((0, 1, 1), (0, 0, 1)),
         ((1, 0, 1), (1, 0, 0)), ((1, 1, 1), (1, 1, 0)), ((0, 1, 1), (0, 1, 0)),
         ((1, 0, 0), (1, 1, 0)), ((1, 1, 0), (0, 1, 0))]
    for p, q in E:
        d.line([iso(*p), iso(*q)], fill=(255, 255, 255, 150), width=int(S * 0.006))

    glow = layer.filter(ImageFilter.GaussianBlur(S * 0.012))
    img = Image.alpha_composite(img, Image.eval(glow, lambda v: v // 2))
    img = Image.alpha_composite(img, layer)
    img = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)

    os.makedirs(OUT_DIR, exist_ok=True)
    img.save(os.path.join(OUT_DIR, "icon_1024.png"))
    json.dump({"images": [{"filename": "icon_1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
               "info": {"author": "xcode", "version": 1}}, open(os.path.join(OUT_DIR, "Contents.json"), "w"), indent=2)
    json.dump({"info": {"author": "xcode", "version": 1}},
              open(os.path.join(OUT_DIR, "..", "Contents.json"), "w"), indent=2)
    print("wrote", os.path.normpath(os.path.join(OUT_DIR, "icon_1024.png")))


if __name__ == "__main__":
    main()
