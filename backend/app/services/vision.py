

# app/services/vision.py
from __future__ import annotations

import os
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np


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
    scale = max_side / m
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    img = _resize_max(bgr, 1600)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 140)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (H * W) * 0.01:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)

        if ar < 1.2 or ar > 6.8:
            continue
        if w < 140 or h < 70:
            continue
        if y < int(0.18 * H):
            continue
        if y > int(0.92 * H):
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:4]  # ✅ 여러 알약도 가능하게 넉넉히

    # 원본 스케일 복원
    h0, w0 = bgr.shape[:2]
    sx = w0 / W
    sy = h0 / H

    out = []
    for x, y, w, h in boxes:
        X = int(x * sx)
        Y = int(y * sy)
        W0 = int(w * sx)
        H0 = int(h * sy)

        pad = int(0.10 * max(W0, H0))
        X = max(0, X - pad)
        Y = max(0, Y - pad)
        X2 = min(w0, X + W0 + 2 * pad)
        Y2 = min(h0, Y + H0 + 2 * pad)
        out.append((X, Y, X2 - X, Y2 - Y))

    # 좌->우 정렬(일관성)
    out.sort(key=lambda r: r[0])
    return out


def _shape_from_contour(bgr_roi: np.ndarray, debug_dir: Optional[str], prefix: str) -> str:
    roi = _resize_max(bgr_roi, 900)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    if debug_dir:
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_mask.png"), th)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "타원형"

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h + 1e-6)

    if debug_dir:
        vis = roi.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_contour.png"), vis)

    if ar >= 2.2:
        return "장방형"
    if 1.3 <= ar < 2.2:
        return "타원형"
    return "원형"


def _label_color_from_hsv(h: float, s: float, v: float) -> str:
    # white
    if s < 45 and v > 170:
        return "하양"
    if v < 60:
        return "기타"

    # OpenCV H: 0~179
    if (h < 10) or (h > 160):
        return "빨강" if s >= 120 else "주황"
    if 10 <= h < 30:
        return "주황"
    if 30 <= h < 45:
        return "노랑"
    if 45 <= h < 85:
        return "연두"
    if 85 <= h < 140:
        return "파랑"
    return "기타"


def _pill_silhouette_mask(bgr_roi: np.ndarray, debug_dir: Optional[str] = None, prefix: str = "vision") -> np.ndarray:
    roi = _resize_max(bgr_roi, 900)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    else:
        mask[:] = 255

    if debug_dir:
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_pillmask.png"), mask)

    return mask


def _dominant_colors(bgr_roi: np.ndarray, debug_dir: Optional[str] = None, prefix: str = "vision", k: int = 1) -> List[str]:
    roi = _resize_max(bgr_roi, 900)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = _pill_silhouette_mask(roi, debug_dir, prefix)
    ys, xs = np.where(mask > 0)

    if len(xs) < 500:
        y0, y1 = int(roi.shape[0] * 0.25), int(roi.shape[0] * 0.75)
        x0, x1 = int(roi.shape[1] * 0.25), int(roi.shape[1] * 0.75)
        sub = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    else:
        sub = hsv[ys, xs].reshape(-1, 3).astype(np.float32)

    if sub.shape[0] > 6000:
        idx = np.random.choice(sub.shape[0], 6000, replace=False)
        sub = sub[idx]

    K = max(1, int(k))
    if K == 1:
        h, s, v = np.mean(sub, axis=0)
        return [_label_color_from_hsv(float(h), float(s), float(v))]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(sub, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=K)
    order = np.argsort(counts)[::-1]

    out = []
    for i in order:
        h, s, v = centers[i]
        out.append(_label_color_from_hsv(float(h), float(s), float(v)))

    uniq = []
    for c in out:
        if c not in uniq and c != "기타":
            uniq.append(c)

    return uniq[:K] if uniq else ["기타"]


def _is_capsule(roi_bgr: np.ndarray) -> bool:
    """
    아주 단순 캡슐 판별:
    - 길이비(aspect) 높고
    - 좌/우 평균 hue 차이가 꽤 나면 2톤 캡슐로 간주
    """
    roi = _resize_max(roi_bgr, 900)
    h, w = roi.shape[:2]
    aspect = w / max(h, 1)
    if aspect < 1.3:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    left = hsv[:, : w // 2]
    right = hsv[:, w // 2 :]

    h_l = float(np.mean(left[..., 0]))
    h_r = float(np.mean(right[..., 0]))

    return abs(h_l - h_r) > 12


def guess_shape_color_multi(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "vision",
) -> List[Dict]:
    """
    반환 예:
    [
      {"roi":(x,y,w,h), "shape":"타원형", "is_capsule":True, "colors":["빨강","파랑"], "color":"빨강+파랑"},
      ...
    ]
    """
    if bgr is None or bgr.size == 0:
        return []

    rois = _find_pill_rois(bgr)
    results: List[Dict] = []

    for idx, (x, y, w, h) in enumerate(rois, start=1):
        roi = bgr[y:y + h, x:x + w]

        if debug_dir:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_roi{idx}.png"), roi)

        shape = _shape_from_contour(roi, debug_dir, f"{debug_prefix}_{idx}")

        is_capsule = _is_capsule(roi)
        k = 2 if (is_capsule or shape in ("타원형", "장방형")) else 1
        colors = _dominant_colors(roi, debug_dir, f"{debug_prefix}_{idx}_color", k=k)

        # ✅ 최종 문자열
        color = "+".join(colors) if colors else None

        results.append(
            {
                "roi": (x, y, w, h),
                "shape": shape,
                "is_capsule": bool(is_capsule),
                "colors": colors,
                "color": color,
            }
        )

    return results
