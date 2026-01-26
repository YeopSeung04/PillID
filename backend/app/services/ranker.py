# app/services/ranker.py
from __future__ import annotations

import re
from typing import Optional, Set


def match_level(score: int) -> str:
    if score >= 120:
        return "HIGH"
    if score >= 70:
        return "MEDIUM"
    return "LOW"


# ----------------------------
# Normalization / Tokenization
# ----------------------------

_TOKEN_RE = re.compile(r"[A-Z0-9\-]{2,}")


def _norm(s: Optional[str]) -> str:
    return (s or "").upper().strip()


def _tokens_list(s: str) -> list[str]:
    """원본 문자열에서 토큰 리스트(순서 유지)가 필요할 때."""
    return _TOKEN_RE.findall(_norm(s))


def _tokens_set(s: str) -> Set[str]:
    """토큰 집합(매칭용)."""
    return set(_tokens_list(s))


def _tok_variants(t: str) -> Set[str]:
    """
    OCR 토큰 변형 흡수:
    - D-W <-> DW
    - 기타 하이픈 제거 버전
    """
    t = _norm(t)
    if not t:
        return set()
    return {t, t.replace("-", "")}


def _tok_weight(t: str) -> int:
    """
    토큰별 가중치.
    - D-W / DW: 강함
    - PAC: 강함
    - 나머지: 중간
    """
    t0 = _norm(t)
    t0_no = t0.replace("-", "")
    if t0 in ("D-W", "DW") or t0_no == "DW":
        return 70
    if t0 == "PAC":
        return 60
    return 35


# ----------------------------
# Color/Shape helpers
# ----------------------------

def _color_aliases(color: Optional[str]) -> Set[str]:
    if not color:
        return set()
    c = color.strip()

    # 표기 흔들림 흡수
    if c == "주황":
        return {"주황", "주황색", "연주황", "적주황"}
    if c == "빨강":
        return {"빨강", "적색", "빨간", "적"}
    if c == "하양":
        return {"하양", "흰색", "백색"}
    if c == "파랑":
        return {"파랑", "청색", "파란"}
    if c == "노랑":
        return {"노랑", "황색", "노란"}
    if c == "연두":
        return {"연두", "녹색", "초록", "초록색"}

    return {c}


def _shape_aliases(shape: Optional[str]) -> Set[str]:
    if not shape:
        return set()
    s = shape.strip()
    # MFDS에서 캡슐은 타원형/장방형이 섞여 들어오는 경우가 많음
    if s == "타원형":
        return {"타원형", "장방형"}
    if s == "장방형":
        return {"장방형", "타원형"}
    return {s}


def _item_has_color(it: dict, guess: Optional[str]) -> bool:
    if not guess:
        return False
    colors = _color_aliases(guess)
    if not colors:
        return False

    c1 = str(it.get("COLOR_CLASS1") or "").strip()
    c2 = str(it.get("COLOR_CLASS2") or "").strip()
    return (c1 in colors) or (c2 in colors)


def _item_has_shape(it: dict, guess: Optional[str]) -> bool:
    if not guess:
        return False
    shapes = _shape_aliases(guess)
    if not shapes:
        return False

    s = str(it.get("DRUG_SHAPE") or "").strip()
    return s in shapes


# ----------------------------
# Main scoring
# ----------------------------

def score_candidate(
    it: dict,
    imprint: str,
    color_guess: Optional[str] = None,
    shape_guess: Optional[str] = None,
) -> int:
    """
    스코어링 정책(실전용, 과하게 죽이지 않기):
    1) 각인 토큰 매칭이 핵심 (PRINT_FRONT/PRINT_BACK 토큰 집합과 교집합)
       - 토큰별 가중치 적용 (D-W/DW, PAC 우선)
       - 하이픈 제거 변형 허용
    2) 색/모양은 '가점'만 (프리필터로도 쓰되, 스코어에서도 보조 신호로 약하게)
       - MFDS는 COLOR_CLASS1/2가 있으니 둘 다 비교
       - shape는 타원형/장방형 alias 허용
    """
    target = _norm(imprint)
    if not target:
        return 0

    # 후보 아이템 각인 토큰 집합 (부분문자열 오탐 방지)
    p_front = _norm(it.get("PRINT_FRONT"))
    p_back = _norm(it.get("PRINT_BACK"))
    p_toks = _tokens_set(p_front) | _tokens_set(p_back)
    if not p_toks:
        # 각인 데이터가 없는 품목은 기본적으로 불리
        # (단, 색/모양만으로도 후보를 열어두고 싶으면 0 리턴하지 말고 아주 작은 점수를 줄 수도 있음)
        p_toks = set()

    score = 0

    # 입력 각인 토큰(순서 유지)
    toks = _tokens_list(target)

    # 토큰별 매칭 (하이픈 변형 포함)
    for t in toks:
        variants = _tok_variants(t)
        if not variants:
            continue

        # token-set 교집합 매칭
        if variants & p_toks:
            score += _tok_weight(t)

    # 색/모양 가점 (너무 강하게 하면 비전 오판에 끌려감)
    if _item_has_shape(it, shape_guess):
        score += 8
    if _item_has_color(it, color_guess):
        score += 8

    return score
