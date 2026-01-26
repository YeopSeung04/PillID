# app/services/vision.py
from __future__ import annotations

import os
from typing import Tuple, Optional, List

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
    """
    알약 ROI만 찾기 (격자/눈금 영역 최대한 제외).
    - 엣지 기반 큰 외곽 물체(알약) 탐지
    - 상단 눈금 영역/하단 워터마크 영역 제거
    """
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

        # 캡슐은 대체로 가로로 김
        if ar < 1.2 or ar > 6.8:
            continue
        if w < 140 or h < 70:
            continue

        # 상단 눈금 제거
        if y < int(0.18 * H):
            continue
        # 하단 워터마크 제거
        if y > int(0.92 * H):
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:2]

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

    return out


def _shape_from_contour(bgr_roi: np.ndarray, debug_dir: Optional[str], prefix: str) -> str:
    """
    ROI 내부에서만 컨투어 기반 형태 추정.
    반환: MFDS 형태 라벨에 맞춰서 "장방형/타원형/원형/기타"
    """
    roi = _resize_max(bgr_roi, 900)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 배경(격자) 제거를 위해 adaptive threshold + close
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    if debug_dir:
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_mask.png"), th)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "타원형"  # 안전 기본값(캡슐이 많음)

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h + 1e-6)

    # 디버그 컨투어
    if debug_dir:
        vis = roi.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_contour.png"), vis)

    # 캡슐: 타원형/장방형이 MFDS에 혼재
    # ar이 크면 장방형 쪽, 아니면 타원형 쪽
    if ar >= 2.2:
        return "장방형"
    if 1.3 <= ar < 2.2:
        return "타원형"

    # 거의 정원형
    return "원형"


def _color_from_roi(bgr_roi: np.ndarray) -> str:
    """
    ROI 평균 색 기반으로 대략적인 색 라벨 추정.
    (두 톤 캡슐은 '주요 색(면적 큰 쪽)'으로 잡음)
    """
    roi = _resize_max(bgr_roi, 900)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 채도 낮고 명도 높으면 흰색 계열
    H, S, V = cv2.split(hsv)
    s_mean = float(np.mean(S))
    v_mean = float(np.mean(V))

    # 하양(흰 캡슐) 우선
    if s_mean < 40 and v_mean > 170:
        return "하양"

    # 붉/주황 계열 판단: Hue가 0 근처 또는 160 이상(빨강), 5~25(주황/노랑)
    h_mean = float(np.mean(H))
    if (h_mean < 10) or (h_mean > 160):
        # 빨강/주황은 채도가 높게 나오는 편
        return "주황" if s_mean < 120 else "빨강"

    if 10 <= h_mean < 30:
        return "주황"
    if 30 <= h_mean < 45:
        return "노랑"
    if 45 <= h_mean < 85:
        return "연두"
    if 85 <= h_mean < 140:
        return "파랑"

    return "기타"


def guess_shape_color(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "vision",
) -> Tuple[Optional[str], Optional[str]]:
    """
    전체 이미지가 아니라 '알약 ROI' 기준으로 shape/color를 추정한다.
    여러 알약이 잡히면, '각인이 있을 확률이 큰(=왼쪽 알약)'로 1순위 ROI 선택:
    - ROI들을 x 기준으로 정렬 후 가장 왼쪽 ROI 사용
    """
    if bgr is None or bgr.size == 0:
        return None, None

    rois = _find_pill_rois(bgr)
    if not rois:
        # ROI 못 잡으면 전체로라도 추정(정확도 낮음)
        shape = _shape_from_contour(bgr, debug_dir, debug_prefix)
        color = _color_from_roi(bgr)
        return shape, color

    rois.sort(key=lambda r: r[0])  # x 오름차순(왼쪽 우선)
    x, y, w, h = rois[0]
    roi = bgr[y : y + h, x : x + w]

    if debug_dir:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_roi.png"), roi)

    shape = _shape_from_contour(roi, debug_dir, debug_prefix)
    color = _color_from_roi(roi)
    return shape, color
