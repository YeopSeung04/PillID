# app/services/imprint_ai.py
from __future__ import annotations

import os, re, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


DEFAULT_ALPHABET = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")

def make_ring_mask(pill_mask: np.ndarray, inner_px: int = 28, outer_px: int = 2) -> np.ndarray:
    """
    알약 마스크에서 '둘레 링'만 남기는 마스크 생성.
    - distanceTransform은 foreground에서 가장 가까운 background까지 거리(=테두리까지 거리)를 줌
    - outer_px: 너무 바깥 edge 노이즈 제거용
    - inner_px: 테두리에서 안쪽으로 얼마나 두껍게 가져올지
    """
    m = (pill_mask > 0).astype(np.uint8) * 255
    if m.ndim != 2:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)  # float32
    ring = ((dist >= float(outer_px)) & (dist <= float(inner_px))).astype(np.uint8) * 255

    # 조금 정리
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    return ring


def decode_radial(dets: List[CharDet], img_w: int, img_h: int) -> str:
    if not dets:
        return ""

    # 1) 중심점(일단 이미지 중심으로, 더 정교하게는 마스크 컨투어 중심)
    cx, cy = img_w / 2.0, img_h / 2.0

    items = []
    for d in dets:
        x0, y0, x1, y1 = d.xyxy
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        ang = math.atan2(my - cy, mx - cx)  # -pi~pi
        items.append((ang, d))

    # 2) 각도 정렬 (시계방향/반시계방향은 상황에 맞게 바꿔도 됨)
    items.sort(key=lambda t: t[0])

    # 3) “정방향 시작점”을 위쪽(각도=-pi/2 근처)으로 회전
    # atan2 기준: 위쪽은 -pi/2 근처
    target = -math.pi / 2
    idx0 = min(range(len(items)), key=lambda i: abs(items[i][0] - target))
    items = items[idx0:] + items[:idx0]

    # 4) 문자열
    s = "".join([d.ch for _, d in items])

    # 5) 중복/노이즈 간단 정리(선택)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)  # AAAA -> AA
    return s

def _load_alphabet_from_yaml(path: str) -> Optional[List[str]]:
    try:
        import yaml
        d = yaml.safe_load(open(path, "r"))
        names = d.get("names")
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        if isinstance(names, list) and len(names) > 1:
            return [str(x) for x in names]
    except Exception:
        return None
    return None



@dataclass
class CharDet:
    ch: str
    conf: float
    xyxy: Tuple[int, int, int, int]


def _letterbox(img: np.ndarray, new_shape: int = 640, color: int = 114):
    h0, w0 = img.shape[:2]
    r = min(new_shape / w0, new_shape / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x0 = x - w / 2
    y0 = y - h / 2
    x1 = x + w / 2
    y1 = y + h / 2
    return np.stack([x0, y0, x1, y1], axis=1)


def _clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w - 1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, w - 1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, h - 1)
    return xyxy


def _nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_th: float) -> List[int]:
    if len(boxes_xyxy) == 0:
        return []
    xywh = []
    for b in boxes_xyxy:
        x0, y0, x1, y1 = b.tolist()
        xywh.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), score_threshold=0.0, nms_threshold=float(iou_th))
    if idxs is None or len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


def _normalize_yolo_output(out: np.ndarray, nc: int) -> np.ndarray:
    """
    Ultralytics YOLOv8 ONNX export output commonly:
      (1, 4+nc, 8400)  e.g. (1, 42, 8400)
    Convert to (N, 4+nc)
    """
    a = np.array(out)
    a = np.squeeze(a)  # (42, 8400) or (8400, 42)
    if a.ndim != 2:
        raise ValueError(f"Unexpected output shape: {out.shape}")

    if a.shape[0] == 4 + nc:
        return a.T  # (8400, 42)
    if a.shape[1] == 4 + nc:
        return a     # (8400, 42)

    # fallback: best-effort
    if a.shape[0] > a.shape[1]:
        return a[:, : 4 + nc]
    return a.T[:, : 4 + nc]

class ImprintCharDetector:
    """
    ✅ 권장: onnxruntime로 추론 (OpenCV DNN은 YOLOv8 ONNX에서 자주 터짐)
    env: IMPRINT_YOLO_ONNX=/path/to/best.onnx
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        conf_th: float = 0.12,
        nms_th: float = 0.45,
        input_size: int = 640,
        prefer_ort: bool = True,
    ):
        self.onnx_path = onnx_path or os.getenv("IMPRINT_YOLO_ONNX")
        yaml_path = os.getenv("IMPRINT_DATA_YAML")
        yaml_alphabet = _load_alphabet_from_yaml(yaml_path) if yaml_path else None
        self.alphabet = alphabet or yaml_alphabet or DEFAULT_ALPHABET

        self.conf_th = float(conf_th)
        self.nms_th = float(nms_th)
        self.input_size = int(input_size)

        self._ort_sess = None
        self._ort_iname = None
        self._ort_oname = None

        self._cv_net = None

        if self.onnx_path and os.path.exists(self.onnx_path):
            if prefer_ort:
                if ort is None:
                    raise RuntimeError("onnxruntime not available. Install onnxruntime to run YOLO ONNX.")
                self._ort_sess = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                self._ort_iname = self._ort_sess.get_inputs()[0].name
                self._ort_oname = self._ort_sess.get_outputs()[0].name
            else:
                # fallback only (not recommended)
                self._cv_net = cv2.dnn.readNetFromONNX(self.onnx_path)

        print("[imprint_ai] backend =",
              "onnxruntime" if self._ort_sess is not None else ("opencv_dnn" if self._cv_net is not None else "none"))
        print("[imprint_ai] onnx_path =", self.onnx_path)

    def is_ready(self) -> bool:
        return (self._ort_sess is not None) or (self._cv_net is not None)

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        if self._ort_sess is not None:
            # onnxruntime expects float32 numpy (1,3,640,640)
            out = self._ort_sess.run([self._ort_oname], {self._ort_iname: blob})[0]
            return out
        # fallback
        self._cv_net.setInput(blob)
        return self._cv_net.forward()

    def infer(self, bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> List[CharDet]:
        if not self.is_ready():
            return []

        img = bgr.copy()
        # 빨간 텍스트 강조 (R 채널 - G 채널)
        b, g, r = cv2.split(img)
        x = cv2.subtract(r, g)  # red ink가 강해짐
        x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        g2 = clahe.apply(gray)
        blur = cv2.GaussianBlur(g2, (0, 0), 1.2)
        sharp = cv2.addWeighted(g2, 1.7, blur, -0.7, 0)
        img = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

        if mask is not None and mask.size == img.shape[:2]:
            bg = cv2.GaussianBlur(img, (0, 0), 3)
            img = np.where(mask[..., None] > 0, img, bg)

        canvas, r, dw, dh = _letterbox(img, self.input_size, 114)

        # IMPORTANT: blobFromImage returns (1,3,H,W) float32
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        blob = blob.astype(np.float32)

        out = self._forward(blob)

        nc = len(self.alphabet)
        pred = _normalize_yolo_output(out, nc=nc)  # (8400, 4+nc)

        xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:].astype(np.float32)
        cls_id = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(len(cls_id)), cls_id]

        keep = conf >= self.conf_th
        if not np.any(keep):
            return []

        xywh = xywh[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        boxes = _xywh_to_xyxy(xywh)  # in letterbox coords

        keep_idx = _nms_xyxy(boxes.copy(), conf.copy(), self.nms_th)
        if not keep_idx:
            return []

        boxes = boxes[keep_idx]
        conf = conf[keep_idx]
        cls_id = cls_id[keep_idx]

        # un-letterbox
        boxes[:, [0, 2]] -= float(dw)
        boxes[:, [1, 3]] -= float(dh)
        boxes /= float(r)

        H0, W0 = img.shape[:2]
        boxes = _clip_xyxy(boxes, W0, H0)

        dets: List[CharDet] = []
        for b, c, k in zip(boxes, conf, cls_id):
            k = int(k)
            if not (0 <= k < len(self.alphabet)):
                continue
            ch = self.alphabet[k]
            x0, y0, x1, y1 = [int(round(v)) for v in b.tolist()]
            if (x1 - x0) < 3 or (y1 - y0) < 3:
                continue
            dets.append(CharDet(ch=ch, conf=float(c), xyxy=(x0, y0, x1, y1)))

        return dets

def decode_detections_radial(
    dets: List[CharDet],
    img_w: int,
    img_h: int,
    conf_th: float = 0.12,
    angle_bin_deg: float = 12.0,
) -> str:
    """
    원형/방사형 각인 디코더:
    - 중심(cx,cy) 기준 각도(theta)로 정렬
    - 시작점은 '가장 위쪽(y 최소)' 문자
    - 각도 버킷(예: 12도)으로 중복 제거(같은 자리 겹검출 방지)
    """
    if not dets:
        return ""

    # 0) 너무 낮은 conf 제거(원형은 노이즈 잘 뜸)
    dets = [d for d in dets if d.conf >= conf_th]
    if not dets:
        return ""

    cx, cy = img_w / 2.0, img_h / 2.0

    items = []
    for d in dets:
        x0, y0, x1, y1 = d.xyxy
        mx = (x0 + x1) * 0.5
        my = (y0 + y1) * 0.5
        ang = math.atan2(my - cy, mx - cx)  # -pi..pi
        items.append((ang, mx, my, d))

    # 1) 각도 정렬(반시계). 시계방향 원하면 reverse 또는 ang 부호 반전.
    items.sort(key=lambda t: t[0])

    # 2) 시작점: 가장 위쪽(y 최소)인 det를 찾아 그 각도 위치로 rotate
    # 2) 시작점: P가 있으면 P, 없으면 가장 위쪽
    p_idx = None
    for i, (_, _, _, d) in enumerate(items):
        if d.ch == "P" and d.conf >= 0.3:
            p_idx = i
            break
    if p_idx is not None:
        items = items[p_idx:] + items[:p_idx]
    else:
        top_i = min(range(len(items)), key=lambda i: items[i][2])
        items = items[top_i:] + items[:top_i]

    # 3) 각도 버킷으로 중복 제거(같은 위치 여러 글자 방지)
    bin_rad = math.radians(angle_bin_deg)
    chosen = {}
    for ang, mx, my, d in items:
        b = int(round(ang / bin_rad))
        prev = chosen.get(b)
        if prev is None or d.conf > prev.conf:
            chosen[b] = d

    # 4) 다시 각도 순서 유지(rotate된 items 기준)
    # rotate된 순서를 최대한 유지하려고 items 순회하며 chosen에 남은 것만 뽑음
    used_bins = set()
    ordered = []
    for ang, mx, my, d in items:
        b = int(round(ang / bin_rad))
        if b in used_bins:
            continue
        if chosen.get(b) is d:
            ordered.append(d)
            used_bins.add(b)

    s = "".join([d.ch for d in ordered]).strip().upper()
    s = re.sub(r"\s+", "", s)
    return s

def decode_detections_to_string(
    dets: List[CharDet],
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    radial: bool = False,
) -> str:

    """
    - y 기준으로 2줄(윗줄/아랫줄) 분리
    - 각 줄은 x 기준 정렬
    - 너무 먼 outlier(우측 경계/노이즈) 제거
    - 근접 중복(거의 같은 위치) 제거
    """
    if radial:
        if img_w is None or img_h is None:
            return ""
        return decode_detections_radial(dets, img_w=img_w, img_h=img_h)

    if not dets:
        return ""

    # 0) 기본 필터: 너무 낮은 conf는 일단 버림 (하이픈은 예외로 살릴 수 있게 별도 처리)
    keep = []
    low_hyphen = []
    for d in dets:
        if d.ch == "-" and d.conf >= 0.12:
            low_hyphen.append(d)
        elif d.conf >= 0.25:
            keep.append(d)
    dets = keep + low_hyphen
    if not dets:
        return ""

    # 1) 중심 좌표 계산
    xs = np.array([(d.xyxy[0] + d.xyxy[2]) * 0.5 for d in dets], dtype=np.float32)
    ys = np.array([(d.xyxy[1] + d.xyxy[3]) * 0.5 for d in dets], dtype=np.float32)

    # 2) outlier 제거: 중앙 x 주변만 남김 (우측 경계 같은 잡음 제거)
    x_med = float(np.median(xs))
    x_mad = float(np.median(np.abs(xs - x_med)) + 1e-6)
    inlier = np.abs(xs - x_med) <= (3.5 * x_mad)
    dets = [d for d, ok in zip(dets, inlier) if ok]
    if not dets:
        return ""

    # 3) y 기준 2줄 분리(간단 k-means 2)
    ys = np.array([(d.xyxy[1] + d.xyxy[3]) * 0.5 for d in dets], dtype=np.float32)
    if len(dets) >= 4:
        # 2-cluster by y
        c1, c2 = float(np.percentile(ys, 30)), float(np.percentile(ys, 70))
        for _ in range(10):
            a = ys < (c1 + c2) / 2
            if a.any():
                c1 = float(ys[a].mean())
            if (~a).any():
                c2 = float(ys[~a].mean())
        top_center = min(c1, c2)
        bot_center = max(c1, c2)
        # 각 줄 할당
        top = [d for d in dets if abs(((d.xyxy[1]+d.xyxy[3])*0.5) - top_center) <= abs(((d.xyxy[1]+d.xyxy[3])*0.5) - bot_center)]
        bot = [d for d in dets if d not in top]
    else:
        # det 적으면 그냥 한 줄로
        top, bot = dets, []

    for d in list(bot):
        if d.ch == "-":
            top.append(d)
            bot.remove(d)

    def dedup_line(line: List[CharDet]) -> List[CharDet]:
        if not line:
            return []
        # x 기준 정렬
        line = sorted(line, key=lambda d: ((d.xyxy[0]+d.xyxy[2])*0.5, -d.conf))
        # (추가) x outlier 컷: line 내부에서 좌우 10~90% 범위만 유지
        cxs = np.array([(d.xyxy[0] + d.xyxy[2]) * 0.5 for d in line], dtype=np.float32)
        lo, hi = np.percentile(cxs, 10), np.percentile(cxs, 90)
        line = [d for d in line if lo - 25 <= ((d.xyxy[0] + d.xyxy[2]) * 0.5) <= hi + 25]

        # 근접 중복 제거
        out = []
        for d in line:
            cx = (d.xyxy[0] + d.xyxy[2]) * 0.5
            if not out:
                out.append(d)
                continue
            px = (out[-1].xyxy[0] + out[-1].xyxy[2]) * 0.5
            if abs(cx - px) < 10:  # 가까우면 conf 높은 것만
                if d.conf > out[-1].conf:
                    out[-1] = d
            else:
                out.append(d)
        return out

    top = dedup_line(top)
    bot = dedup_line(bot)

    def _post_fix_imprint(top_s: str, bot_s: str) -> tuple[str, str]:
        top_s = re.sub(r"\s+", "", top_s).upper()
        bot_s = re.sub(r"\s+", "", bot_s).upper()

        # VV -> W (자주 나옴)
        top_s = top_s.replace("VV", "W")

        # 문자 혼동 보정(일단 1회 치환)
        bot_s = bot_s.replace("S", "C")  # C/S
        bot_s = bot_s.replace("0", "O")  # 0/O
        bot_s = bot_s.replace("1", "I")  # 1/I

        # 윗줄 중복 D 정리
        top_s = re.sub(r"^D{2,}", "D", top_s)

        return top_s, bot_s

    top_s = "".join([d.ch for d in top]).strip().upper()
    bot_s = "".join([d.ch for d in bot]).strip().upper()

    top_s, bot_s = _post_fix_imprint(top_s, bot_s)

    if bot_s:
        return f"{top_s} {bot_s}".strip()
    return top_s

