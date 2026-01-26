import cv2
import numpy as np

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
