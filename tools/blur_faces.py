"""Blur faces in photos/videos before anything leaves the Mac (Apple Vision via tools/faces, on-device).

    .venv/bin/python tools/blur_faces.py marble_inputs        # every area_*/NN.jpg and *.mp4, in place
Small faces are found by running the detector on 2x-upscaled overlapping tiles.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID, OVER, UP = 3, 0.15, 2.0


def detect(images):
    """images: list of BGR arrays -> list of [x0, y0, x1, y1] pixel boxes per image."""
    tmp = tempfile.mkdtemp()
    tiles = []
    for k, im in enumerate(images):
        H, W = im.shape[:2]
        tw, th = W / GRID, H / GRID
        for i in range(GRID):
            for j in range(GRID):
                x0, y0 = max(0, int((j - OVER) * tw)), max(0, int((i - OVER) * th))
                x1, y1 = min(W, int((j + 1 + OVER) * tw)), min(H, int((i + 1 + OVER) * th))
                t = cv2.resize(im[y0:y1, x0:x1], None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
                name = f"{k:05d}_{i}{j}.jpg"
                cv2.imwrite(os.path.join(tmp, name), t, [cv2.IMWRITE_JPEG_QUALITY, 92])
                tiles.append((name, k, x0, y0, x1 - x0, y1 - y0))
        cv2.imwrite(os.path.join(tmp, f"{k:05d}_full.jpg"), im)
        tiles.append((f"{k:05d}_full.jpg", k, 0, 0, W, H))
    out = subprocess.run([os.path.join(HERE, "faces"), tmp], capture_output=True, text=True, check=True).stdout
    by = {t[0]: t for t in tiles}
    boxes = [[] for _ in images]
    for ln in out.splitlines():
        r = json.loads(ln)
        _, k, x0, y0, w, h = by[os.path.basename(r["path"])]
        for a, b, c, d in r["faces"]:
            boxes[k].append([x0 + a * w, y0 + b * h, x0 + c * w, y0 + d * h])
    shutil.rmtree(tmp)
    return boxes


def blur(im, boxes):
    for x0, y0, x1, y1 in boxes:
        p = 0.35 * max(x1 - x0, y1 - y0)
        a, b = int(max(0, x0 - p)), int(max(0, y0 - p))
        c, d = int(min(im.shape[1], x1 + p)), int(min(im.shape[0], y1 + p))
        if c > a and d > b:
            im[b:d, a:c] = cv2.GaussianBlur(im[b:d, a:c], (0, 0), max(6, (c - a) / 4))
    return len(boxes)


def main(root):
    photos = sorted(glob.glob(os.path.join(root, "area_*", "*.jpg")))
    ims = [cv2.imread(p) for p in photos]
    n = 0
    for p, im, bx in zip(photos, ims, detect(ims)):
        if bx:
            n += blur(im, bx)
            cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 93])
            print(f"  {p}: {len(bx)} face(s)")
    print(f"photos: {n} faces blurred in {len(photos)} images")
    for v in sorted(glob.glob(os.path.join(root, "*.mp4")) + glob.glob(os.path.join(root, "area_*", "*.mp4"))):
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        n = 0
        for i0 in range(0, len(frames), 100):
            chunk = frames[i0:i0 + 100]
            for f, bx in zip(chunk, detect(chunk)):
                n += blur(f, bx)
        H, W = frames[0].shape[:2]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                              f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt",
                              "yuv420p", "-movflags", "+faststart", v + ".tmp.mp4"], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        p.wait()
        os.replace(v + ".tmp.mp4", v)
        print(f"{v}: {n} face boxes blurred over {len(frames)} frames")


if __name__ == "__main__":
    main(sys.argv[1])
