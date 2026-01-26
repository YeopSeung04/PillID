# app/services/ranker.py
from __future__ import annotations

import re


def match_level(score: int) -> str:
    if score >= 120:
        return "HIGH"
    if score >= 70:
        return "MEDIUM"
    return "LOW"


def _norm(s: str | None) -> str:
    return (s or "").upper().strip()


def _tokens(s: str) -> list[str]:
    return re.findall(r"[A-Z0-9\-]{2,}", _norm(s))


def score_candidate(it: dict, imprint: str, color_guess: str | None, shape_guess: str | None) -> int:
    """
    간단 스코어:
    - imprint 토큰이 PRINT_FRONT/PRINT_BACK에 포함되면 가점
    - shape/color 일치 가점 (너무 강하게 0으로 죽이지 않도록 '가점'만)
    """
    p1 = _norm(it.get("PRINT_FRONT"))
    p2 = _norm(it.get("PRINT_BACK"))
    target = _norm(imprint)

    if not target:
        return 0

    score = 0

    # 토큰 매칭 (중요)
    toks = _tokens(target)
    for t in toks:
        if t in p1 or t in p2:
            # D-W 같은 토큰은 매우 강하므로 가중치 크게
            if t in ("D-W", "DW"):
                score += 70
            elif t in ("PAC",):
                score += 60
            else:
                score += 35

    # 보조: shape/color는 "가점"만
    if shape_guess and it.get("DRUG_SHAPE") and str(it.get("DRUG_SHAPE")).strip() == shape_guess:
        score += 10
    if color_guess and it.get("COLOR_CLASS1") and str(it.get("COLOR_CLASS1")).strip() == color_guess:
        score += 10

    return score
