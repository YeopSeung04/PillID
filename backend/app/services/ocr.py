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

    return _normalize_text(" ".join(parts))


def _crop_center_band(bgr: np.ndarray) -> np.ndarray:
    """
    중앙 밴드만 사용: 반사/워터마크/배경 영향 줄이기
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
    """
    붉은/갈색 계열 각인 강조용(일부 캡슐에서 먹힘)
    """
    b, g, r = cv2.split(bgr)
    red = cv2.subtract(r, g)
    red = cv2.GaussianBlur(red, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(red)


<<<<<<< Updated upstream
=======
def _prep_deboss_edges(bgr: np.ndarray) -> np.ndarray:
    """
    ✅ 음각/엠보싱(흰 알약 RX 같은) 전용 전처리:
    CLAHE + Sobel magnitude로 '경계'를 살려서 OCR에 넣는다.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g = clahe.apply(g)

    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu threshold
    _, th = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


>>>>>>> Stashed changes
def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    (fallback용) OCR에서만 쓰는 간단 ROI: 알약 덩어리 찾기.
    ※ 메인 파이프라인에선 vision.py가 ROI를 이미 주니까 이건 거의 안 쓰게 됨.
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

        if ar < 1.1 or ar > 7.0:
            continue
        if w < 120 or h < 60:
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:3]

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

<<<<<<< Updated upstream
=======

def extract_imprint_text_roi(
    roi_bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
) -> str:
    """
    ✅ 단일 ROI 전용 OCR
    - color -> upsharp(gray) -> clahe(gray) -> red -> deboss(edge) -> deboss(invert)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return ""

    crop = _crop_center_band(roi_bgr)

    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_roi.png"), crop)

    # 1) 컬러
    t = _ocr_run(crop)
    if t:
        return t

    # 2) upsharp gray
    ug = _prep_upsharp_gray(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_upsharp.png"), ug)
    t = _ocr_run(ug)
    if t:
        return t

    # 3) clahe gray
    g = _prep_gray_clahe(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_clahe.png"), g)
    t = _ocr_run(g)
    if t:
        return t

    # 4) red emphasis
    r = _prep_red_emphasis(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_red.png"), r)
    t = _ocr_run(r)
    if t:
        return t

    # 5) deboss/emboss edges
    d = _prep_deboss_edges(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_deboss.png"), d)
    t = _ocr_run(d)
    if t:
        return t

    # 6) deboss invert (케이스에 따라 글자 대비가 반대)
    di = 255 - d
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_deboss_inv.png"), di)
    return _ocr_run(di)

>>>>>>> Stashed changes

def extract_imprint_text(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
) -> str:
    """
    (legacy) 전체 이미지에서 OCR하려는 경우.
    지금 메인 파이프라인에선 vision이 ROI를 주니까 거의 안 써도 됨.
    """
    if bgr is None or bgr.size == 0:
        return ""

    rois = _find_pill_rois(bgr)

    # ROI 못 잡으면 fallback (중앙 밴드)
    if not rois:
        return extract_imprint_text_roi(bgr, debug_dir=debug_dir, debug_prefix=f"{debug_prefix}_fallback" if debug_prefix else None)

    results: List[str] = []
    for i, (x, y, w, h) in enumerate(rois, start=1):
        crop0 = bgr[y:y + h, x:x + w]
        t = extract_imprint_text_roi(
            crop0,
            debug_dir=debug_dir,
            debug_prefix=f"{debug_prefix}_pill{i}" if debug_prefix else None,
        )
        if t:
            results.append(t)

    return _normalize_text(" ".join(results))
