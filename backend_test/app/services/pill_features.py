# app/services/pill_features.py
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _label_color_from_hsv(h: float, s: float, v: float) -> str:
    """HSV 평균값을 한국어 라벨로 변환."""
    if s < 45 and v > 170:
        return "하양"
    if v < 60:
        return "기타"

    # red wrap-around
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


def guess_color_from_mask(bgr: np.ndarray, mask: np.ndarray) -> Optional[str]:
    """마스크 내부 HSV 평균으로 색 추정."""
    if bgr is None or bgr.size == 0 or mask is None or mask.size == 0:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[ys, xs].mean(axis=0)
    c = _label_color_from_hsv(float(h), float(s), float(v))
    return c if c != "기타" else None


def guess_shape_from_mask(mask: np.ndarray) -> Optional[str]:
    """마스크 컨투어 기반으로 모양(원형/타원형/장방형) 추정."""
    if mask is None or mask.size == 0:
        return None

    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 400:
        return None

    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h + 1e-6)
    peri = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (peri * peri + 1e-6)

    if ar >= 2.2:
        return "장방형"
    if ar < 1.25 and circ >= 0.72:
        return "원형"
    return "타원형"


def is_capsule_from_mask(mask: np.ndarray, shape: Optional[str]) -> bool:
    """간단 캡슐 판단: 타원/장방 + 가로가 충분히 김."""
    if mask is None or mask.size == 0:
        return False
    h, w = mask.shape[:2]
    if h <= 0:
        return False
    if shape not in ("타원형", "장방형"):
        return False
    return (w / float(h)) > 1.3


def tight_crop_to_mask(bgr: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """마스크 bbox로 더 타이트하게 자르기."""
    if bgr is None or bgr.size == 0 or mask is None or mask.size == 0:
        return bgr, mask
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return bgr, mask
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = int(pad_ratio * max(x1 - x0 + 1, y1 - y0 + 1))
    H, W = mask.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return bgr[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]
