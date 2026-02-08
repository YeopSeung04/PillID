# app/services/imprint_roi.py
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np

def extract_imprint_roi(bgr: np.ndarray, pill_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    알약 크롭(bgr) + 알약 마스크(pill_mask)에서
    '각인(텍스트) 가능 영역'만 자동으로 추출해서 리턴.
    """
    H, W = bgr.shape[:2]
    m = (pill_mask > 0).astype(np.uint8) * 255

    # 1) 배경 억제: 알약 영역만 유지
    roi = bgr.copy()
    roi[m == 0] = (255, 255, 255)

    # 2) 그레이 + CLAHE로 대비 올리기
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # 3) black-hat: 밝은 바탕 위 어두운(또는 상대적으로 어두운) 텍스트를 강조
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)

    # 4) 이진화 + 정리
    bh = cv2.GaussianBlur(bh, (3, 3), 0)
    th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), 2)

    # 알약 마스크 내부만
    th = cv2.bitwise_and(th, m)

    # 5) 가장 큰 텍스트 블롭 근처로 박스
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # 못 찾으면 중앙부 fallback
        x0, x1 = int(W * 0.05), int(W * 0.95)
        y0, y1 = int(H * 0.20), int(H * 0.85)
        return bgr[y0:y1, x0:x1], m[y0:y1, x0:x1]

    # 여러 글자 덩어리를 하나로 합치는 느낌으로 union box
    xs, ys, xe, ye = W, H, 0, 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 60:  # 노이즈 컷
            continue
        x, y, w, h = cv2.boundingRect(c)
        xs = min(xs, x); ys = min(ys, y)
        xe = max(xe, x + w); ye = max(ye, y + h)

    if xe <= xs or ye <= ys:
        x0, x1 = int(W * 0.05), int(W * 0.95)
        y0, y1 = int(H * 0.20), int(H * 0.85)
        return bgr[y0:y1, x0:x1], m[y0:y1, x0:x1]

    # 패딩
    pad = int(0.12 * max(xe - xs, ye - ys))
    x0 = max(0, xs - pad); y0 = max(0, ys - pad)
    x1 = min(W, xe + pad); y1 = min(H, ye + pad)

    return bgr[y0:y1, x0:x1], m[y0:y1, x0:x1]
