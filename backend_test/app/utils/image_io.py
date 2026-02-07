# app/utils/image_io.py
from __future__ import annotations

import numpy as np
import cv2
from fastapi import UploadFile


async def read_upload_to_bgr(file: UploadFile, save_path: str | None = None) -> np.ndarray:
    """
    UploadFile -> OpenCV BGR 이미지로 디코딩.
    save_path가 있으면 원본 바이트를 그대로 저장.
    """
    raw = await file.read()
    if save_path:
        try:
            with open(save_path, "wb") as f:
                f.write(raw)
        except Exception:
            pass

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imdecode failed")
    return img
