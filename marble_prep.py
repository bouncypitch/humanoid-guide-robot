"""Prepare World Labs Marble inputs from an RGB-D walkthrough: upright, people blurred, sharp.

    python marble_prep.py captures/session_20261024-1347 marble_inputs --areas 4

Writes:
  marble_inputs/walkthrough.mp4         whole walkthrough, upright, people blurred (Marble video input)
  marble_inputs/area_K/NN.jpg           up to 8 sharp, spread-out photos per area (Marble multi-image input)
  marble_inputs/area_K/clip.mp4         short clip of just that area
  marble_inputs/areas.png               floor-plan-free overview: camera positions coloured by area
  marble_inputs/manifest.json           which frames went where, for aligning Marble's output later
People are blurred with the iPhone's person masks (dilated, feathered) before anything leaves the Mac.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rgbd_ingest as ri  # noqa: E402
from landmarks import orientation  # noqa: E402


def sharpness(rgb):
    import cv2

    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def clean_image(f, rot):
    im = Image.fromarray(f.rgb)
    if f.mask is not None and f.mask.max() > 0:
        m = Image.fromarray(f.mask).resize(im.size, Image.BILINEAR)
        m = m.filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.GaussianBlur(12))
        blurred = im.filter(ImageFilter.GaussianBlur(28))
        im = Image.composite(blurred, im, m)
    return im.rotate(rot, expand=True)


def farthest_point(xy, k):
    idx = [int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))]
    d = np.linalg.norm(xy - xy[idx[0]], axis=1)
    while len(idx) < min(k, len(xy)):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(xy - xy[i], axis=1))
    return idx


def write_video(images, path, fps=12):
    w, h = images[0].size
    w, h = w - w % 2, h - h % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for im in images:
        p.stdin.write(np.asarray(im.convert("RGB").resize((w, h))).tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--areas", type=int, default=4)
    ap.add_argument("--photos", type=int, default=8)
    ap.add_argument("--video-every", type=int, default=2, help="use every Nth keyframe in the whole video")
    a = ap.parse_args()
    root = ri.unzip_session(a.session) if a.session.endswith(".zip") else ri.find_session_root(a.session)
    frames = list(ri.iter_session(root))
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(frames)} frames")

    pos = np.array([f.T[:2, 3] for f in frames])
    sharp = np.array([sharpness(f.rgb) for f in frames])
    people = np.array([(f.mask > 0).mean() if f.mask is not None else 0.0 for f in frames])
    good = (sharp > np.percentile(sharp, 40)) & (people < 0.08)

    # whole walkthrough video
    vid = [clean_image(f, orientation(f)) for f in frames[::a.video_every]]
    write_video(vid, os.path.join(a.out, "walkthrough.mp4"))
    print(f"walkthrough.mp4: {len(vid)} frames @12 fps")

    # areas = k-means on camera xy (time-contiguity not required)
    from sklearn.cluster import KMeans

    lab = KMeans(n_clusters=a.areas, n_init=10, random_state=0).fit(pos).labels_
    manifest = {"session": os.path.abspath(root), "areas": {}}
    for k in range(a.areas):
        ids = np.where((lab == k) & good)[0]
        if len(ids) < 2:
            ids = np.where(lab == k)[0]
        d = os.path.join(a.out, f"area_{k}")
        os.makedirs(d, exist_ok=True)
        # spread over position AND heading: farthest-point on (x, y, 1.5*cos h, 1.5*sin h)
        hd = np.array([np.arctan2(frames[i].T[1, 2], frames[i].T[0, 2]) for i in ids])
        feat = np.column_stack([pos[ids], 1.5 * np.cos(hd), 1.5 * np.sin(hd)])
        pick = [ids[j] for j in farthest_point(feat, a.photos)]
        pick.sort()
        for n, i in enumerate(pick):
            clean_image(frames[i], orientation(frames[i])).save(os.path.join(d, f"{n:02d}.jpg"), quality=93)
        span = np.where(lab == k)[0]
        write_video([clean_image(frames[i], orientation(frames[i])) for i in span[::2]], os.path.join(d, "clip.mp4"))
        manifest["areas"][f"area_{k}"] = {"frames": [int(frames[i].index) for i in pick],
                                          "center_xy": pos[lab == k].mean(0).round(2).tolist(),
                                          "n_frames": int((lab == k).sum())}
        print(f"area_{k}: {len(pick)} photos, {int((lab == k).sum())} frames, centre {pos[lab == k].mean(0).round(1)}")
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pos[:, 0], pos[:, 1], c=lab, cmap="tab10", s=6)
    for k, v in manifest["areas"].items():
        ax.annotate(k, v["center_xy"], fontsize=12, weight="bold")
    ax.set_aspect("equal")
    ax.set_title("Walkthrough camera positions by Marble area")
    fig.savefig(os.path.join(a.out, "areas.png"), dpi=110)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
