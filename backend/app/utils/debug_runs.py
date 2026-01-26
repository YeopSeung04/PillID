# app/utils/debug_runs.py
from __future__ import annotations

import os
import secrets
from datetime import datetime


def make_run_dir(base_dir: str = "debug_runs") -> str:
    """
    요청(실행) 1회당 폴더 1개 생성
    예: debug_runs/20260127_034119_68df69
    """
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = secrets.token_hex(3)  # 6 chars
    run_dir = os.path.join(base_dir, f"{ts}_{rid}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
