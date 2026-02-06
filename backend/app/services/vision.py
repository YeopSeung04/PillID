# app/services/vision.py
from __future__ import annotations

import os
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np


# -------------------------------------------------
# utils
# -------------------------------------------------
def _safe_imwrite(path: str, img: np.ndarray) -> None:
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass


def _resize_max(bgr: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    s = max_side / m
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# -------------------------------------------------
# ROI detection (tray-free + watershed)
# -------------------------------------------------
def _split_watershed(bgr_roi: np.ndarray, bin_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    mask = (bin_mask > 0).astype(np.uint8) * 255

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, fg = cv2.threshold(dist, 0.45, 1.0, cv2.THRESH_BINARY)
    fg = (fg * 255).astype(np.uint8)

    bg = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 2)
    unknown = cv2.subtract(bg, fg)

    n, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    markers = cv2.watershed(bgr_roi.copy(), markers)

    boxes = []
    for lbl in range(2, n + 2):
        ys, xs = np.where(markers == lbl)
        if len(xs) < 200:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w > 30 and h > 30:
            boxes.append((x0, y0, w, h))
    return boxes


def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    KPIC/약학정보원 스타일(파란 배경 + 워터마크/치수선) 대응 ROI 검출
    - 파란 배경(HSV hue 85~140, saturation 높은 영역) 제거
    - 남은 영역에서 알약(대개 분홍/주황/빨강 계열) 연결요소 추출
    """
    img = _resize_max(bgr, 1600)
    H, W = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    # 1) 파란 배경 마스크 (KPIC 배경이 대체로 이 구간)
    blue_bg = ((hh >= 85) & (hh <= 140) & (ss >= 35)).astype(np.uint8) * 255

    # 2) 알약 후보: "파란 배경이 아닌" + 너무 어둡지 않은 픽셀
    obj = cv2.bitwise_and(cv2.bitwise_not(blue_bg), (vv > 40).astype(np.uint8) * 255)

    # 3) 얇은 선(치수선/텍스트) 제거용 open -> close
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k_open, iterations=1)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k_close, iterations=2)

    cnts, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes_local: List[Tuple[int, int, int, int]] = []
    min_area = (H * W) * 0.01  # KPIC 이미지는 알약이 크게 찍힘

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 80 or h < 80:
            continue

        # 너무 길쭉한 것(치수선/바) 제거
        ar = w / float(h + 1e-6)
        if ar > 5.5 or ar < 0.18:
            continue

        boxes_local.append((x, y, w, h))

    boxes_local.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes_local = boxes_local[:6]

    # 원본 스케일로 복원 + 패딩
    h0, w0 = bgr.shape[:2]
    sx = w0 / W
    sy = h0 / H

    out = []
    for x, y, w, h in boxes_local:
        X = int(x * sx); Y = int(y * sy)
        W0 = int(w * sx); H0 = int(h * sy)

        pad = int(0.10 * max(W0, H0))
        X = max(0, X - pad); Y = max(0, Y - pad)
        X2 = min(w0, X + W0 + 2 * pad)
        Y2 = min(h0, Y + H0 + 2 * pad)
        out.append((X, Y, X2 - X, Y2 - Y))

    out.sort(key=lambda r: r[0])
    return out



# -------------------------------------------------
# Mask / shape / color
# -------------------------------------------------
def _pill_silhouette_mask(bgr_roi: np.ndarray, debug_dir: Optional[str], prefix: str) -> np.ndarray:
    roi = _resize_max(bgr_roi, 900)
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    _, th_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def score(th):
        h, w = th.shape
        c = th[int(.3*h):int(.7*h), int(.3*w):int(.7*w)].mean()
        b = np.concatenate([th[:int(.12*h), :].ravel(), th[-int(.12*h):, :].ravel()]).mean()
        return c - b

    th = th_bin if score(th_bin) >= score(th_inv) else th_inv

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 2)

    mask = np.zeros_like(th)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

    if debug_dir:
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_pillmask.png"), mask)
    return mask


def _shape_from_mask(mask: np.ndarray) -> str:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "타원형"
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h + 1e-6)
    peri = cv2.arcLength(c, True)
    circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri + 1e-6)
    if ar >= 2.2:
        return "장방형"
    if ar < 1.25 and circ >= 0.72:
        return "원형"
    return "타원형"


def _label_color_from_hsv(h, s, v) -> str:
    def _me_:

    if s < 45 and v > 170:
        return "하양"
    if v < 60:
        return "기타"
    if h < 10 or h > 160:
        return "빨강" if s >= 120 else "주황"
    if h < 30:
        return "주황"
    if h < 45:
        return "노랑"
    if h < 85:
        return "연두"
    if h < 140:
        return "파랑"
    return "기타"


def _dominant_colors_capsule_halves(bgr_roi: np.ndarray, mask: np.ndarray) -> List[str]:
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, w = mask.shape
    mid = w // 2

    def mean(region):
        ys, xs = np.where(region > 0)
        if len(xs) < 200:
            return None
        return hsv[ys, xs].mean(axis=0)

    out = []
    for m in (mask[:, :mid], mask[:, mid:]):
        v = mean(m)
        if v is not None:
            out.append(_label_color_from_hsv(*v))

    return list(dict.fromkeys([c for c in out if c != "기타"])) or ["기타"]


def _rotate_upright(bgr_roi: np.ndarray, mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr_roi, mask, 0.0
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    angle = rect[2] + (90 if rect[1][0] < rect[1][1] else 0)
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return (
        cv2.warpAffine(bgr_roi, M, (w, h), flags=cv2.INTER_CUBIC),
        cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST),
        angle
    )


def _tight_crop(bgr_roi: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return bgr_roi, mask
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.1 * max(x1-x0, y1-y0))
    return (
        bgr_roi[y0-pad:y1+pad, x0-pad:x1+pad],
        mask[y0-pad:y1+pad, x0-pad:x1+pad],
    )


# -------------------------------------------------
# public API
# -------------------------------------------------
def guess_shape_color_multi(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "vision",
) -> List[Dict]:
    rois = _find_pill_rois(bgr)
    results = []

    for i, (x, y, w, h) in enumerate(rois, 1):
        roi0 = bgr[y:y+h, x:x+w]
        roi = _resize_max(roi0, 900)

        mask = _pill_silhouette_mask(roi, debug_dir, f"{debug_prefix}_{i}")
        rot, rmask, angle = _rotate_upright(roi, mask)
        crop, cmask = _tight_crop(rot, rmask)

        shape = _shape_from_mask(cmask)
        is_capsule = shape in ("타원형", "장방형") and crop.shape[1] / crop.shape[0] > 1.3

        colors = (
            _dominant_colors_capsule_halves(crop, cmask)
            if is_capsule else
            [_label_color_from_hsv(*cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[cmask > 0].mean(axis=0))]
        )

        results.append({
            "roi": (x, y, w, h),
            "shape": shape,
            "is_capsule": is_capsule,
            "colors": colors,
            "color": "+".join(colors),
            "roi_img": crop,
            "angle": angle,
        })

    return results
