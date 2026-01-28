# app/services/vision.py
from __future__ import annotations

import os
from typing import Tuple, Optional, List

import cv2
import numpy as np

def _split_watershed(bgr_roi: np.ndarray, bin_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    붙은 알약을 watershed로 분리해서 (x,y,w,h) 리스트 반환 (ROI 로컬 좌표)
    """
    mask = (bin_mask > 0).astype(np.uint8) * 255

    # distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # sure foreground
    _, sure_fg = cv2.threshold(dist_norm, 0.45, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    # sure background
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sure_bg = cv2.dilate(mask, k, iterations=2)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    # watershed needs 3ch image
    ws_img = bgr_roi.copy()
    markers = cv2.watershed(ws_img, markers)

    boxes = []
    for lbl in range(2, num_labels + 2):
        ys, xs = np.where(markers == lbl)
        if len(xs) < 200:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        if w < 30 or h < 30:
            continue
        boxes.append((int(x0), int(y0), int(w), int(h)))

    return boxes


def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    ✅ 트레이 가정 제거.
    - 어두운 배경(천/테이블) 위의 알약을 '밝기(V채널) 기반'으로 이진화
    - 컨투어/컴포넌트로 ROI 추출
    - 큰 덩어리는 watershed로 분리(붙은 알약)
    """
    img = _resize_max(bgr, 1600)
    H, W = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # 1) Otsu로 밝은 물체(알약) 마스크
    _, th = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) 모폴로지로 노이즈 제거/구멍 메움
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k1, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k1, iterations=2)

    # 3) 컨투어 찾기
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes_local: List[Tuple[int, int, int, int]] = []

    min_area = (H * W) * 0.002  # 화면 대비 최소 알약 면적(0.2%)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue

        # 덩어리가 너무 크면(붙은 알약 가능) -> ROI내 watershed 시도
        # 기준: 면적이 큰데 종횡비가 이상하거나, 그냥 area가 큼
        ar = w / float(h + 1e-6)
        if area > (H * W) * 0.02 or ar > 3.2 or ar < 0.35:
            roi = img[y:y + h, x:x + w]
            roi_mask = th[y:y + h, x:x + w]
            split_boxes = _split_watershed(roi, roi_mask)
            if split_boxes:
                for sx, sy, sw, sh in split_boxes:
                    boxes_local.append((x + sx, y + sy, sw, sh))
                continue  # 원래 큰 박스는 버리고 분리 결과만 사용

        boxes_local.append((x, y, w, h))

    # 4) 큰 것부터 정렬 + 상위 N개(필요시 늘려도 됨)
    boxes_local.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes_local = boxes_local[:6]

    # 5) 원본 스케일로 복원 + 패딩
    h0, w0 = bgr.shape[:2]
    sx = w0 / W
    sy = h0 / H

    out = []
    for x, y, w, h in boxes_local:
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

    # 좌->우 정렬(보기 좋게)
    out.sort(key=lambda r: r[0])
    return out

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


<<<<<<< Updated upstream
def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    알약 ROI만 찾기 (격자/눈금 영역 최대한 제외).
    - 엣지 기반 큰 외곽 물체(알약) 탐지
    - 상단 눈금 영역/하단 워터마크 영역 제거
    """
=======
# def _find_pill_rois(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    boxes = boxes[:2]
=======
    boxes = boxes[:4]
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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

=======
    out.sort(key=lambda r: r[0])
    return out


def _shape_from_mask(mask: np.ndarray) -> str:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "타원형"
>>>>>>> Stashed changes
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h + 1e-6)

<<<<<<< Updated upstream
    # 디버그 컨투어
    if debug_dir:
        vis = roi.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_contour.png"), vis)
=======
    peri = float(cv2.arcLength(c, True))
    circ = (4.0 * np.pi * area / (peri * peri + 1e-6)) if peri > 0 else 0.0
>>>>>>> Stashed changes

    # 캡슐: 타원형/장방형이 MFDS에 혼재
    # ar이 크면 장방형 쪽, 아니면 타원형 쪽
    if ar >= 2.2:
        return "장방형"
<<<<<<< Updated upstream
    if 1.3 <= ar < 2.2:
        return "타원형"

    # 거의 정원형
    return "원형"
=======
    if (ar < 1.25) and (circ >= 0.72):
        return "원형"
    return "타원형"
>>>>>>> Stashed changes


def _color_from_roi(bgr_roi: np.ndarray) -> str:
    roi = _resize_max(bgr_roi, 900)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 알약 픽셀만 남기기: 채도/명도 기반 마스크
    H, S, V = cv2.split(hsv)

    # 너무 어둡거나(그림자) 너무 채도 높은 배경(격자) 일부를 배제
    mask = (V > 60).astype(np.uint8) * 255

    # 흰색(채도 낮음)도 포함해야 하니까 S로 너무 강하게 거르면 안 됨
    # 대신 '파란 격자'는 H가 85~140에 많이 몰림 -> 그 구간을 약하게 제외
    blue_bg = ((H >= 85) & (H <= 140) & (S > 40)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(blue_bg))

    # 마스크 정리
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 마스크가 너무 작으면 fallback
    if cv2.countNonZero(mask) < 500:
        mask = np.ones_like(mask) * 255

    # 마스크 영역 평균 계산
    h_vals = H[mask > 0]
    s_vals = S[mask > 0]
    v_vals = V[mask > 0]

    s_mean = float(np.mean(s_vals))
    v_mean = float(np.mean(v_vals))

    # 흰색 판정 (흰 반쪽 때문에 필요)
    if s_mean < 45 and v_mean > 170:
        return "하양"

    h_mean = float(np.mean(h_vals))

    if (h_mean < 10) or (h_mean > 160):
        return "빨강" if s_mean >= 120 else "주황"
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
    k = 2 if shape in ("타원형", "장방형") else 1
    colors = _dominant_colors(roi, debug_dir, debug_prefix, k=k)
    color = "+".join(colors) # "하양+주황" 같은 형태
    return shape, color

def _label_color_from_hsv(h: float, s: float, v: float) -> str:
    if s < 45 and v > 170:
        return "하양"
    if v < 60:
        return "기타"

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

<<<<<<< Updated upstream
def _pill_silhouette_mask(bgr_roi: np.ndarray, debug_dir: Optional[str]=None, prefix: str="vision") -> np.ndarray:
=======

def _pill_silhouette_mask(bgr_roi: np.ndarray, debug_dir: Optional[str] = None, prefix: str = "vision") -> np.ndarray:
    """
    Robust silhouette mask:
    - Otsu THRESH_BINARY / THRESH_BINARY_INV 둘 다 만들고
    - "center는 pill, border는 background" 기준으로 더 좋은 쪽 선택
    """
>>>>>>> Stashed changes
    roi = _resize_max(bgr_roi, 900)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

<<<<<<< Updated upstream
    # 밝은 배경(격자)에서 알약이 더 어둡게 잡히는 경우가 많아서 INV+OTSU가 보통 유리
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
=======
    _, th_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
>>>>>>> Stashed changes

    def score_mask(th: np.ndarray) -> float:
        h, w = th.shape[:2]
        c = th[int(0.3*h):int(0.7*h), int(0.3*w):int(0.7*w)]
        border = np.concatenate([
            th[:int(0.12*h), :].ravel(),
            th[-int(0.12*h):, :].ravel(),
            th[:, :int(0.12*w)].ravel(),
            th[:, -int(0.12*w):].ravel(),
        ])
        # center는 흰색(알약), border는 검정(배경)일수록 점수↑
        return float(c.mean() - border.mean())

    th = th_bin if score_mask(th_bin) >= score_mask(th_inv) else th_inv

    # morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    else:
        # 그래도 실패하면 ROI 전체가 아니라 중앙 타원 fallback(배경 덜 먹게)
        h, w = mask.shape[:2]
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.45)), 0, 0, 360, 255, -1)

    if debug_dir:
        _safe_imwrite(os.path.join(debug_dir, f"{prefix}_pillmask.png"), mask)

    return mask

<<<<<<< Updated upstream
def _dominant_colors(bgr_roi: np.ndarray, debug_dir: Optional[str]=None, prefix: str="vision", k: int=2) -> list[str]:
=======
def _dominant_colors_capsule_halves(bgr_roi: np.ndarray, mask: np.ndarray) -> List[str]:
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, w = mask.shape[:2]
    mid = w // 2

    def mean_hsv(region_mask: np.ndarray, region_hsv: np.ndarray):
        ys, xs = np.where(region_mask > 0)
        if len(xs) < 200:
            return None
        sub = region_hsv[ys, xs].reshape(-1, 3).astype(np.float32)
        return np.mean(sub, axis=0)

    left_m = np.zeros_like(mask); left_m[:, :mid] = mask[:, :mid]
    right_m = np.zeros_like(mask); right_m[:, mid:] = mask[:, mid:]

    ml = mean_hsv(left_m, hsv)
    mr = mean_hsv(right_m, hsv)

    out = []
    if ml is not None:
        out.append(_label_color_from_hsv(float(ml[0]), float(ml[1]), float(ml[2])))
    if mr is not None:
        out.append(_label_color_from_hsv(float(mr[0]), float(mr[1]), float(mr[2])))

    # 중복 제거 + 기타 제거
    uniq = []
    for c in out:
        if c != "기타" and c not in uniq:
            uniq.append(c)
    return uniq if uniq else ["기타"]


def _dominant_colors(bgr_roi: np.ndarray, debug_dir: Optional[str] = None, prefix: str = "vision", k: int = 1) -> List[str]:
>>>>>>> Stashed changes
    roi = _resize_max(bgr_roi, 900)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask = _pill_silhouette_mask(roi, debug_dir, prefix)

    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        # 마스크 실패 시 fallback: ROI 중앙부라도 사용
        y0, y1 = int(roi.shape[0]*0.25), int(roi.shape[0]*0.75)
        x0, x1 = int(roi.shape[1]*0.25), int(roi.shape[1]*0.75)
        sub = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    else:
        sub = hsv[ys, xs].reshape(-1, 3).astype(np.float32)

    # 다운샘플(속도/안정)
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

    # 중복 제거
    uniq = []
    for c in out:
        if c not in uniq:
            uniq.append(c)

<<<<<<< Updated upstream
    return uniq[:K]
=======
    return uniq[:K] if uniq else ["기타"]


def _rotate_upright_by_mask(bgr_roi: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rotate ROI so that the main pill axis becomes horizontal."""
    if bgr_roi is None or bgr_roi.size == 0:
        return bgr_roi, mask, 0.0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr_roi, mask, 0.0

    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = float(rect[2])
    (h, w) = mask.shape[:2]

    if rect[1][0] < rect[1][1]:
        angle = angle + 90.0

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(bgr_roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rmask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot, rmask, angle


def _tight_crop_by_mask(bgr_roi: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return bgr_roi, mask
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    pad = int(pad_ratio * max(x1 - x0 + 1, y1 - y0 + 1))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)
    return bgr_roi[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]


def _is_capsule_masked(roi_bgr: np.ndarray, mask: np.ndarray) -> bool:
    """2톤 캡슐 간단 판별(마스크 내부 픽셀만 사용)."""
    roi = roi_bgr
    h, w = roi.shape[:2]
    aspect = w / max(h, 1)
    if aspect < 1.3:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mid = w // 2
    lm = mask[:, :mid] > 0
    rm = mask[:, mid:] > 0
    if lm.sum() < 200 or rm.sum() < 200:
        return False

    h_l = float(np.mean(hsv[:, :mid][lm][:, 0]))
    h_r = float(np.mean(hsv[:, mid:][rm][:, 0]))
    return abs(h_l - h_r) > 12


def guess_shape_color_multi(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "vision",
) -> List[Dict]:
    if bgr is None or bgr.size == 0:
        return []

    rois = _find_pill_rois(bgr)
    results: List[Dict] = []

    for idx, (x, y, w, h) in enumerate(rois, start=1):
        roi0 = bgr[y:y + h, x:x + w]

        if debug_dir:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_roi{idx}.png"), roi0)

        # 1) 마스크 -> 회전 정규화 -> 타이트 크롭
        roi = _resize_max(roi0, 900)
        mask = _pill_silhouette_mask(roi, debug_dir, f"{debug_prefix}_{idx}")
        rot, rmask, angle = _rotate_upright_by_mask(roi, mask)
        crop, cmask = _tight_crop_by_mask(rot, rmask, pad_ratio=0.10)

        if debug_dir:
            _safe_imwrite(os.path.join(debug_dir, f"{debug_prefix}_{idx}_upright.png"), crop)

        # 2) shape/capsule/color는 정규화 crop 기준
        shape = _shape_from_mask(cmask)
        is_capsule = _is_capsule_masked(crop, cmask)

        if is_capsule:
            colors = _dominant_colors_capsule_halves(crop, cmask)
        else:
            colors = _dominant_colors(crop, debug_dir, f"{debug_prefix}_{idx}_color", k=1)

        color = "+".join(colors) if colors else None

        results.append(
            {
                "roi": (x, y, w, h),
                "shape": shape,
                "is_capsule": bool(is_capsule),
                "colors": colors,
                "color": color,
                # ✅ OCR에 바로 쓰라고 제공
                "roi_img": crop,
                "angle": float(angle),
            }
        )

    return results
>>>>>>> Stashed changes
