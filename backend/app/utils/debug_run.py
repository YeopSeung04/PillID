# app/utils/debug_runs.py
from __future__ import annotations

import os
import uuid
from datetime import datetime


def make_run_dir(base_dir: str = "debug_runs") -> str:
    """
    debug_runs/YYYYMMDD_HHMMSS_xxxxxx 형태로 폴더 생성 후 경로 반환.
    """
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:6]
    run_dir = os.path.join(base_dir, f"{ts}_{suf}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir.replace("\\", "/")
