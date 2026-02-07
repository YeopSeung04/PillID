# app/services/imprint_ai.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# 학습 시 클래스 순서 고정 (필요한 문자만 두면 됨)
DEFAULT_ALPHABET = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")


@dataclass
class CharDet:
    ch: str
    conf: float
    xyxy: Tuple[int, int, int, int]


def _letterbox(img: np.ndarray, new_shape: int = 640, color: int = 114):
    """
    Ultralytics 스타일 letterbox
    Returns: canvas, scale, (dw, dh)
    """
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
    # xywh: (N,4) center-based
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
    """
    cv2.dnn.NMSBoxes는 xywh를 받기 때문에 변환해서 사용
    """
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


def _as_array(out) -> np.ndarray:
    """
    cv2.dnn forward 결과가 tuple/list로 오는 경우 흡수.
    """
    if isinstance(out, (list, tuple)):
        # 보통 (1, N, 4+nc) 하나만 옴. 여러개면 첫 번째 사용.
        out = out[0]
    return np.array(out)


def _parse_ultralytics_output(out: np.ndarray, nc: int) -> np.ndarray:
    """
    Ultralytics YOLO ONNX 흔한 출력들을 (N, 4+nc) 형태로 정규화.
    지원:
      - (1, N, 4+nc)
      - (1, 4+nc, N)
      - (N, 4+nc)
      - (4+nc, N)
    반환: (N, 4+nc)
    """
    a = out
    a = np.squeeze(a)

    if a.ndim == 3:
        # (1, N, D) or (1, D, N)
        a = np.squeeze(a, axis=0)

    if a.ndim != 2:
        raise ValueError(f"Unsupported YOLO output ndim={a.ndim}, shape={out.shape}")

    # now 2D: either (N, D) or (D, N)
    if a.shape[1] == 4 + nc:
        return a
    if a.shape[0] == 4 + nc:
        return a.T

    # 어떤 export는 obj 포함 (5+nc)도 있음 → nc 추정/흡수
    if a.shape[1] >= 4 + nc:
        return a[:, :4 + nc]
    if a.shape[0] >= 4 + nc:
        return a.T[:, :4 + nc]

    raise ValueError(f"Cannot normalize YOLO output. got shape={a.shape}, nc={nc}")


class ImprintCharDetector:
    """
    ONNX YOLO 문자 검출기 (실제 동작 버전)

    요구:
      - ultralytics YOLO로 문자(클래스=DEFAULT_ALPHABET) 학습
      - export onnx
      - env: IMPRINT_YOLO_ONNX=/path/to/model.onnx

    가정(ultralytics export 기본):
      - output: xywh(center) + class scores (no obj) 형태가 일반적
      - obj 포함 형태면 자동으로 최대값 기반으로라도 동작하도록 처리
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        alphabet: Optional[List[str]] = None,
        conf_th: float = 0.35,
        nms_th: float = 0.45,
        input_size: int = 640,
        use_cuda: bool = False,
    ):
        self.onnx_path = onnx_path or os.getenv("IMPRINT_YOLO_ONNX")
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.conf_th = float(conf_th)
        self.nms_th = float(nms_th)
        self.input_size = int(input_size)

        self._net = None
        if self.onnx_path and os.path.exists(self.onnx_path):
            self._net = cv2.dnn.readNetFromONNX(self.onnx_path)
            if use_cuda:
                try:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                except Exception:
                    # CUDA 없으면 조용히 CPU로
                    pass

    def is_ready(self) -> bool:
        return self._net is not None

    def infer(self, bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> List[CharDet]:
        if not self.is_ready():
            return []

        img = bgr.copy()

        # mask 있으면 배경 흐림으로 텍스트 대비 강화(반사/워터마크 억제)
        if mask is not None and mask.size == img.shape[:2]:
            bg = cv2.GaussianBlur(img, (0, 0), 3)
            img = np.where(mask[..., None] > 0, img, bg)

        canvas, r, dw, dh = _letterbox(img, self.input_size, 114)

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self._net.setInput(blob)
        out = self._net.forward()
        out = _as_array(out)

        nc = len(self.alphabet)
        pred = _parse_ultralytics_output(out, nc=nc)  # (N, 4+nc) or (N, 5+nc) 잘리게 처리

        if pred.shape[1] < 4 + 1:
            return []

        # pred: [x,y,w,h, scores...]
        xywh = pred[:, :4].astype(np.float32)

        scores_all = pred[:, 4:].astype(np.float32)  # (N, nc) 혹은 (N, nc+obj) 꼴도 잘려 들어올 수 있음
        if scores_all.shape[1] < 1:
            return []

        # 어떤 모델은 obj가 포함(5+nc)이라면:
        # - scores_all[:,0]이 obj일 수도 있음 -> 완벽히 구분 못 해도
        #   "최대 class score" 기반으로 동작하게 처리
        cls_scores = scores_all
        cls_id = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(len(cls_id)), cls_id]

        keep = conf >= self.conf_th
        if not np.any(keep):
            return []

        xywh = xywh[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        boxes_in = _xywh_to_xyxy(xywh)

        # NMS in letterbox space
        keep_idx = _nms_xyxy(boxes_in.copy(), conf.copy(), self.nms_th)
        if not keep_idx:
            return []

        boxes_in = boxes_in[keep_idx]
        conf = conf[keep_idx]
        cls_id = cls_id[keep_idx]

        # letterbox -> original image coords
        # undo padding, then divide by scale
        boxes_in[:, [0, 2]] -= float(dw)
        boxes_in[:, [1, 3]] -= float(dh)
        boxes_in /= float(r)

        H0, W0 = img.shape[:2]
        boxes_in = _clip_xyxy(boxes_in, W0, H0)

        dets: List[CharDet] = []
        for b, c, k in zip(boxes_in, conf, cls_id):
            k = int(k)
            if 0 <= k < len(self.alphabet):
                ch = self.alphabet[k]
            else:
                continue

            x0, y0, x1, y1 = [int(round(v)) for v in b.tolist()]
            if (x1 - x0) < 3 or (y1 - y0) < 3:
                continue

            dets.append(CharDet(ch=ch, conf=float(c), xyxy=(x0, y0, x1, y1)))

        return dets


def decode_detections_to_string(dets: List[CharDet]) -> str:
    """
    문자 bbox들을 좌→우 정렬해서 문자열 복원.
    (줄바꿈은 MVP에서는 무시. 필요하면 y기준 클러스터링 추가)
    """
    if not dets:
        return ""

    # 1) x-center 기준 정렬
    dets = sorted(dets, key=lambda d: ((d.xyxy[0] + d.xyxy[2]) * 0.5, -d.conf))

    # 2) 너무 가까운 중복 박스 제거(같은 위치에 2개 뜨는 경우)
    cleaned: List[CharDet] = []
    for d in dets:
        if not cleaned:
            cleaned.append(d)
            continue
        px = (cleaned[-1].xyxy[0] + cleaned[-1].xyxy[2]) * 0.5
        cx = (d.xyxy[0] + d.xyxy[2]) * 0.5
        if abs(cx - px) < 4:  # 거의 같은 위치면 conf 높은 것만
            if d.conf > cleaned[-1].conf:
                cleaned[-1] = d
        else:
            cleaned.append(d)

    s = "".join([d.ch for d in cleaned]).strip().upper()
    return s
