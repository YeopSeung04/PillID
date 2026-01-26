# app/services/ocr.py
from __future__ import annotations

import os
import re
from typing import Optional, List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

_OCR = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False,
)


# 기존 extract_imprint_text(...)를 그대로 재사용
def extract_imprint_text_roi(
    bgr_roi: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
) -> str:
    """
    ROI 단위 OCR.
    내부적으로 기존 extract_imprint_text를 그대로 호출.
    """
    return extract_imprint_text(bgr_roi, debug_dir=debug_dir, debug_prefix=debug_prefix)



def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.upper().strip()
    s = re.sub(r"[^A-Z0-9\.\-\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_imwrite(path: str, img: np.ndarray) -> None:
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass


def _resize_max(bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _ocr_run(img: np.ndarray, conf_th: float = 0.35) -> str:
    try:
        result = _OCR.ocr(img, cls=True)
    except Exception:
        return ""

    parts: List[str] = []
    for block in result or []:
        for entry in block or []:
            try:
                txt = str(entry[1][0])
                conf = float(entry[1][1])
            except Exception:
                continue
            if txt and conf >= conf_th:
                parts.append(txt)

    # PaddleOCR은 파편들을 붙여주는 편이 OCR에 유리한 경우가 많음
    merged = _normalize_text(" ".join(parts))
    return merged


def _crop_center_band(bgr: np.ndarray) -> np.ndarray:
    """
    눈금/워터마크 방지: 중앙 밴드만 사용
    """
    h, w = bgr.shape[:2]
    y1 = int(h * 0.10)
    y2 = int(h * 0.92)
    return bgr[y1:y2, :]


def _prep_gray_clahe(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(g)


def _prep_upsharp_gray(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    sharp = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    return sharp


def _prep_red_emphasis(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    red = cv2.subtract(r, g)
    red = cv2.GaussianBlur(red, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(red)

def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    OCR용 ROI: 알약 덩어리만 잡기.
    """
    img = _resize_max(bgr, 1800)
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

        if ar < 1.2 or ar > 6.5:
            continue
        if w < 140 or h < 70:
            continue

        if y < int(0.18 * H):
            continue
        if y > int(0.90 * H):
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

        pad = int(0.12 * max(W0, H0))
        X = max(0, X - pad)
        Y = max(0, Y - pad)
        X2 = min(w0, X + W0 + 2 * pad)
        Y2 = min(h0, Y + H0 + 2 * pad)
        out.append((X, Y, X2 - X, Y2 - Y))

    return out

# app/services/ocr.py 하단에 추가

def extract_imprint_text_roi(
    roi_bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
) -> str:
    """
    단일 ROI에서만 OCR 수행 (다중 알약 섞임 방지)
    """
    crop = _crop_center_band(roi_bgr)

    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_roi.png"), crop)

    # 1) 컬러
    t = _ocr_run(crop)
    if t:
        return t

    # 2) upsharp
    ug = _prep_upsharp_gray(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_upsharp.png"), ug)
    t = _ocr_run(ug)
    if t:
        return t

    # 3) clahe
    g = _prep_gray_clahe(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_clahe.png"), g)
    t = _ocr_run(g)
    if t:
        return t

    # 4) red
    r = _prep_red_emphasis(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_red.png"), r)
    return _ocr_run(r)


def extract_imprint_text(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
) -> str:
    """
    전략:
    1) 알약 ROI 찾기
    2) ROI마다 원본 컬러 -> upsharp(gray) -> gray+clahe -> red 강조 순으로 OCR
    3) debug_dir 있으면 전부 해당 폴더에 저장
    """
    if bgr is None or bgr.size == 0:
        return ""

    rois = _find_pill_rois(bgr)

    def _p(name: str) -> Optional[str]:
        if not (debug_dir and debug_prefix):
            return None
        return os.path.join(debug_dir, f"{debug_prefix}_{name}.png")

    # ROI 못 잡으면 fallback (중앙 밴드)
    if not rois:
        crop = _crop_center_band(bgr)
        p = _p("roi_fallback")
        if p:
            _safe_imwrite(p, crop)

        t = _ocr_run(crop)
        if t:
            return t

        ug = _prep_upsharp_gray(crop)
        p = _p("prep_upsharp_fallback")
        if p:
            _safe_imwrite(p, ug)
        t = _ocr_run(ug)
        if t:
            return t

        g = _prep_gray_clahe(crop)
        p = _p("prep_clahe_fallback")
        if p:
            _safe_imwrite(p, g)
        t = _ocr_run(g)
        if t:
            return t

        r = _prep_red_emphasis(crop)
        p = _p("prep_red_fallback")
        if p:
            _safe_imwrite(p, r)
        return _ocr_run(r)

    results: List[str] = []

    for i, (x, y, w, h) in enumerate(rois, start=1):
        crop0 = bgr[y : y + h, x : x + w]
        crop0 = _crop_center_band(crop0)

        if debug_dir and debug_prefix:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_roi_pill{i}.png"), crop0)

        # 1) 원본 컬러
        t1 = _ocr_run(crop0)
        if t1:
            results.append(t1)
            continue

        # 2) upsharp(gray)
        ug = _prep_upsharp_gray(crop0)
        if debug_dir and debug_prefix:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_upsharp_pill{i}.png"), ug)
        t2 = _ocr_run(ug)
        if t2:
            results.append(t2)
            continue

        # 3) gray+clahe
        g = _prep_gray_clahe(crop0)
        if debug_dir and debug_prefix:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_clahe_pill{i}.png"), g)
        t3 = _ocr_run(g)
        if t3:
            results.append(t3)
            continue

        # 4) red emphasis
        r = _prep_red_emphasis(crop0)
        if debug_dir and debug_prefix:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_prep_red_pill{i}.png"), r)
        t4 = _ocr_run(r)
        if t4:
            results.append(t4)
            continue

    return _normalize_text(" ".join(results))
