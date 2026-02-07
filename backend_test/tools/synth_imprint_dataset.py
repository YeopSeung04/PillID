"""합성 각인(문자 bbox) 데이터 생성기 (YOLO char-detector 학습용).

- 실제 알약 실루엣/배경이 없어도 MVP 학습 가능하게 '텍스트 + 음각/반사'만 합성.
- 출력:
  out/
    images/train/*.jpg
    labels/train/*.txt   (YOLO 포맷: cls cx cy w h)

사용 예)
  python synth_imprint_dataset.py --out out --n 20000 --img 512

NOTE:
- 알파벳/클래스 순서는 imprint_ai.DEFAULT_ALPHABET과 동일해야 함.
- 나중에 실제 캡슐 크롭 RGBA 위에 오버레이로 합성하면 더 강해짐.
"""

from __future__ import annotations

import os
import random
import string
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ALPHABET = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")


def _rand_token() -> str:
    # MFDS 각인에서 흔한 길이 2~6
    L = random.choice([2, 3, 3, 4, 4, 5, 6])
    # 하이픈이 들어갈 확률 조금
    chars = random.choices(ALPHABET[:-2], k=L)  # '-' '.' 제외
    s = "".join(chars)
    if L >= 3 and random.random() < 0.18:
        k = random.randint(1, L - 2)
        s = s[:k] + "-" + s[k:]
    return s


def _emboss_effect(img: np.ndarray, strength: float) -> np.ndarray:
    """음각/엠보싱 비슷한 느낌: 그라디언트 + 조명 스폿."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # emboss-like blend
    out = cv2.addWeighted(cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR), strength, img, 1.0, 0)

    # specular highlight spot
    h, w = g.shape
    cx = random.randint(int(0.25 * w), int(0.75 * w))
    cy = random.randint(int(0.25 * h), int(0.75 * h))
    rad = random.randint(int(0.12 * min(h, w)), int(0.30 * min(h, w)))
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    spot = np.clip(1.0 - d2 / float(rad * rad), 0.0, 1.0)
    spot = (spot * random.uniform(10, 45)).astype(np.float32)
    out = np.clip(out.astype(np.float32) + spot[..., None], 0, 255).astype(np.uint8)
    return out


def _render_sample(img_size: int) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """(image, yolo_labels)"""
    H = W = img_size

    # pill-ish background
    base = np.full((H, W, 3), random.randint(180, 245), dtype=np.uint8)

    # slight gradient
    grad = np.linspace(random.randint(-12, 0), random.randint(0, 12), W).astype(np.int16)
    base = np.clip(base.astype(np.int16) + grad[None, :, None], 0, 255).astype(np.uint8)

    # random pill color tint
    tint = np.array([
        random.randint(-10, 10),
        random.randint(-10, 10),
        random.randint(-10, 10),
    ], dtype=np.int16)
    base = np.clip(base.astype(np.int16) + tint[None, None, :], 0, 255).astype(np.uint8)

    text = _rand_token()

    # choose font
    font = random.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ])

    scale = random.uniform(1.2, 2.4) * (img_size / 512.0)
    thickness = random.randint(2, 4)

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # center-ish
    x = random.randint(max(5, (W - tw) // 2 - 40), min(W - tw - 5, (W - tw) // 2 + 40))
    y = random.randint(max(th + 10, (H + th) // 2 - 30), min(H - 10, (H + th) // 2 + 30))

    # render on separate layer to compute bboxes
    layer = np.zeros_like(base)

    # random "ink" color close to background -> deboss 느낌
    ink = random.randint(60, 140)
    color = (ink, ink, ink)

    cv2.putText(layer, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # slight rotation
    ang = random.uniform(-18, 18)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    layer = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # blend into base
    out = cv2.addWeighted(base, 1.0, layer, random.uniform(0.45, 0.90), 0)

    # emboss/lighting
    out = _emboss_effect(out, strength=random.uniform(0.25, 0.75))

    # noise
    noise = np.random.normal(0, random.uniform(2.0, 7.0), (H, W, 3)).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # compute per-char bboxes (rough): render each char separately and find nonzero bbox
    labels = []
    x_cursor = x
    for ch in text:
        if ch not in ALPHABET:
            continue
        (cw, chh), _ = cv2.getTextSize(ch, font, scale, thickness)
        char_layer = np.zeros((H, W), dtype=np.uint8)
        cv2.putText(char_layer, ch, (x_cursor, y), font, scale, 255, thickness, cv2.LINE_AA)
        char_layer = cv2.warpAffine(char_layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        ys, xs = np.where(char_layer > 10)
        if len(xs) > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            cls = ALPHABET.index(ch)
            cx = (x0 + x1) / 2 / W
            cy = (y0 + y1) / 2 / H
            bw = (x1 - x0 + 1) / W
            bh = (y1 - y0 + 1) / H
            labels.append((cls, cx, cy, bw, bh))
        # advance cursor (simple)
        x_cursor += int(cw * random.uniform(0.85, 1.05))

    # augment: blur occasionally
    if random.random() < 0.35:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out_synth")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--split", type=float, default=0.9)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "images/val").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/val").mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        img, labels = _render_sample(args.img)
        is_train = (random.random() < args.split)
        split = "train" if is_train else "val"

        stem = f"synth_{i:07d}"
        img_path = out / f"images/{split}/{stem}.jpg"
        lbl_path = out / f"labels/{split}/{stem}.txt"

        cv2.imwrite(str(img_path), img)

        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls, cx, cy, bw, bh in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if (i + 1) % 500 == 0:
            print(f"generated {i+1}/{args.n}")

    # dataset yaml helper
    yaml = out / "dataset.yaml"
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {out.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(ALPHABET)}\n")
        f.write("names: [" + ", ".join([f"'{c}'" for c in ALPHABET]) + "]\n")

    print("done:", out)


if __name__ == "__main__":
    main()
