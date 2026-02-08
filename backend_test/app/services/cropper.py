# app/services/cropper.py

import cv2
import numpy as np


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """마스크에서 가장 큰 연결요소만 남긴다."""
    if mask is None or mask.size == 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    # 0은 background
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (labels == idx).astype(np.uint8) * 255
    return out

def auto_crop_pill(bgr: np.ndarray) -> np.ndarray:
    """
    간단 컨투어 기반 크롭.
    실패하면 원본 반환.
    """
    h, w = bgr.shape[:2]
    if h < 80 or w < 80:
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지 → 닫기 연산 → 컨투어
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr

    # 가장 그럴듯한(면적 큰) 컨투어
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < (h * w) * 0.02:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)

        # 너무 화면 전체면 패스
        if cw * ch > (h * w) * 0.9:
            continue

        pad = int(0.08 * max(cw, ch))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad); y1 = min(h, y + ch + pad)
        crop = bgr[y0:y1, x0:x1]
        if crop.size > 0:
            return crop

    return bgr


def crop_and_segment_pill(
    bgr: np.ndarray,
    pad_ratio: float = 0.10,
    grabcut_iters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 알약 가정용: 크롭 + 배경 제거(마스크) + RGBA 생성

    Returns:
      crop_bgr: (H,W,3)
      mask:     (H,W) 0/255
      rgba:     (H,W,4) alpha 포함
    """
    if bgr is None or bgr.size == 0:
        return bgr, np.zeros((0, 0), np.uint8), np.zeros((0, 0, 4), np.uint8)

    crop = auto_crop_pill(bgr)
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        m = np.zeros((h, w), np.uint8)
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = m
        return crop, m, rgba

    # GrabCut 초기 rect: 바깥 패딩 제외 영역
    pad = int(pad_ratio * max(h, w))
    x0, y0 = pad, pad
    x1, y1 = max(pad + 1, w - pad), max(pad + 1, h - pad)
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop, gc_mask, rect, bgd, fgd, grabcut_iters, cv2.GC_INIT_WITH_RECT)
        mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        # grabcut 실패하면 Otsu fallback
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # foreground가 밝은 경우가 많음 -> 중앙 평균으로 방향 결정
        ch, cw = th.shape
        c = th[int(.3*ch):int(.7*ch), int(.3*cw):int(.7*cw)].mean()
        b = np.concatenate([th[:int(.12*ch), :].ravel(), th[-int(.12*ch):, :].ravel()]).mean()
        mask = th if (c - b) > 0 else cv2.bitwise_not(th)

    # 후처리: 구멍 메우고 가장 큰 덩어리만
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = _largest_cc(mask)

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = mask
    return crop, mask, rgba
