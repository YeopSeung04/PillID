# features.py

import cv2
import numpy as np
from typing import Optional

def guess_color(bgr: np.ndarray) -> Optional[str]:
    # 중앙부 평균 HSV로 대충 색 분류
    h, w = bgr.shape[:2]
    y0, y1 = int(h*0.25), int(h*0.75)
    x0, x1 = int(w*0.25), int(w*0.75)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[...,0]))
    s_mean = float(np.mean(hsv[...,1]))
    v_mean = float(np.mean(hsv[...,2]))

    # 채도 낮고 밝으면 흰색 계열
    if s_mean < 35 and v_mean > 170:
        return "WHITE"
    if v_mean < 60:
        return "DARK"

    # 대략적 hue 기반
    if 15 <= h_mean < 35:
        return "YELLOW"
    if 35 <= h_mean < 85:
        return "GREEN"
    if 85 <= h_mean < 130:
        return "BLUE"
    if 130 <= h_mean < 170:
        return "PURPLE"
    # 빨강은 0 근처도 포함
    if h_mean < 15 or h_mean >= 170:
        return "RED"

    return None

def guess_shape(bgr: np.ndarray) -> Optional[str]:
    # 컨투어 원형도/종횡비로 대충
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 배경 반전 케이스 대응
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 200:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / max(h, 1)

    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / max(peri * peri, 1)

    # 원형도 높으면 원형
    if circularity > 0.75:
        return "ROUND"
    # 종횡비 큰 장방형
    if aspect > 1.6 or aspect < 0.62:
        return "OBLONG"
    return "OVAL"
