from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ✅ 토큰 정규식: 파일 최상단에 딱 1번만
_TOK_RE = re.compile(r"[A-Z0-9\-\.]{2,}")

_OCR = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False,
)


def _safe_imwrite(path: str, img: np.ndarray) -> None:
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.upper().strip()
    s = re.sub(r"[^A-Z0-9\.\-\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def _prep_deboss_edges(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g = clahe.apply(g)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # ✅ binary로 자르지 말고 mag 그대로 반환
    return mag


def extract_imprint_text_roi(
    roi_bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
) -> str:
    """단일 ROI OCR: color -> upsharp -> clahe -> red -> deboss -> deboss_inv"""
    if roi_bgr is None or roi_bgr.size == 0:
        return ""

    crop = _crop_center_band(roi_bgr)

    if debug_dir and debug_prefix:
        os.makedirs(debug_dir, exist_ok=True)
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

    # 5) deboss edges
    d = _prep_deboss_edges(crop)
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_deboss.png"), d)
    t = _ocr_run(d, conf_th=0.20)
    if t:
        return t

    # 6) deboss invert
    di = 255 - d
    if debug_dir and debug_prefix:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_deboss_inv.png"), di)
    return _ocr_run(di, conf_th=0.20)


# -------------------------
# round / radial unwrap OCR
# -------------------------
def _rotate_keep(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _apply_white_bg(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[mask == 0] = (255, 255, 255)
    return out


def _score_text(t: str) -> tuple[int, int]:
    toks = _TOK_RE.findall((t or "").upper())
    return (len(toks), len((t or "").replace(" ", "")))


def extract_imprint_text_round_unwrap_ocr(
    crop_bgr: np.ndarray,
    pill_mask: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: Optional[str] = None,
    save_best_only: bool = True,  # ✅ 기본: best만 저장
    max_debug_saves: int = 3,  # ✅ 최대로 저장할 개수
) -> str:
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    if pill_mask is None or pill_mask.size == 0:
        return ""

    m = (pill_mask > 0).astype(np.uint8) * 255

    # 배경 하얗게
    base = crop_bgr.copy()
    base[m == 0] = (255, 255, 255)

    # 음각 에지 + 반전
    edge = _prep_deboss_edges(base)
    edge_inv = 255 - edge

    h, w = edge.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    maxR = min(cx, cy)

    # ✅ sweep 과도하게 하지 말고 현실적으로 줄여라
    angles = list(range(0, 360, 20))  # 18개
    bands = [(0.50, 0.98), (0.55, 0.98), (0.60, 0.98), (0.65, 0.98)]  # 4개
    flips = [False, True]
    invs = [False, True]

    best_t = ""
    best_sc = (-1, -1)
    best_img = None
    best_tag = ""
    debug_saved = 0

    def score_text(t: str) -> tuple[int, int]:
        toks = _TOK_RE.findall((t or "").upper())
        return (len(toks), len((t or "").replace(" ", "")))

    def try_src(src: np.ndarray, tag: str):
        nonlocal best_t, best_sc, best_img, best_tag, debug_saved

        for ang in angles:
            rot = _rotate_keep(src, ang)
            polar = cv2.warpPolar(
                rot,
                (900, int(maxR)),  # 폭 넉넉히
                (cx, cy),
                maxR,
                cv2.WARP_POLAR_LINEAR,
            )

            for flip in flips:
                P = cv2.flip(polar, 1) if flip else polar
                ph, pw = P.shape[:2]

                for bi, (rs, re) in enumerate(bands):
                    y0 = int(ph * rs)
                    y1 = int(ph * re)
                    strip = P[y0:y1, :]
                    strip = cv2.resize(strip, (pw, 260), interpolation=cv2.INTER_CUBIC)

                    # Otsu + close
                    _, th = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    th = cv2.morphologyEx(
                        th,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
                        1,
                    )

                    for inv in invs:
                        img_ocr = (255 - th) if inv else th
                        t = _ocr_run(img_ocr, conf_th=0.18)

                        # 너무 짧은 쓰레기 컷
                        if not t or len(t.replace(" ", "")) < 3:
                            continue

                        sc = score_text(t)
                        if sc > best_sc:
                            best_sc = sc
                            best_t = t
                            best_img = img_ocr
                            best_tag = f"{tag}_a{ang}_f{int(flip)}_b{bi}_{'inv' if inv else 'nor'}"

                        # ✅ 디버그는 원하면 소량만 저장
                        if debug_dir and debug_prefix and (not save_best_only) and debug_saved < max_debug_saves:
                            _safe_imwrite(
                                os.path.join(
                                    debug_dir,
                                    f"{debug_prefix}_unwrap_{tag}_a{ang}_f{int(flip)}_b{bi}_{'inv' if inv else 'nor'}.png",
                                ),
                                img_ocr,
                            )
                            debug_saved += 1

    try_src(edge, "edge")
    try_src(edge_inv, "edgeinv")

    # ✅ best만 저장
    if debug_dir and debug_prefix and save_best_only and best_img is not None:
        _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_unwrap_best_{best_tag}.png"), best_img)

    return best_t
