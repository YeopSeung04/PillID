# app/services/ranker.py
from __future__ import annotations

import re
from typing import Optional, Set, List


def match_level(score: int) -> str:
    if score >= 120:
        return "HIGH"
    if score >= 70:
        return "MEDIUM"
    return "LOW"


def _norm(s: Optional[str]) -> str:
    return (s or "").upper().strip()


# 토큰: A-Z/0-9/하이픈, 길이 2+
_TOKEN_RE = re.compile(r"[A-Z0-9\-]{2,}")


def _tokens(s: str) -> List[str]:
    """외부(main.py)에서도 쓸 토큰화 함수."""
    toks = _TOKEN_RE.findall(_norm(s))
    out = []
    for t in toks:
        # "-W" 같은 쓰레기 토큰 제거
        if t.startswith("-") or t.endswith("-"):
            continue
        out.append(t)
    return out


def _tokens_set(s: str) -> Set[str]:
    return set(_tokens(s))


def _tok_variants(t: str) -> Set[str]:
    t = _norm(t)
    return {t, t.replace("-", "")}


def _tok_weight(t: str) -> int:
    t0 = _norm(t)
    t0_no = t0.replace("-", "")
    if t0 in ("D-W", "DW") or t0_no == "DW":
        return 70
    if t0 == "PAC":
        return 60
    return 35


def _split_color_guess(color_guess: Optional[str]) -> Set[str]:
    """
    "빨강+파랑" / "빨강,파랑" / "빨강 파랑" 등을 set으로.
    """
    if not color_guess:
        return set()
    parts = re.split(r"[+,/ ]+", color_guess.strip())
    return {p.strip() for p in parts if p.strip()}


def _color_aliases_one(c: str) -> Set[str]:
    c = c.strip()
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


def _item_has_color(it: dict, color_guess: Optional[str]) -> bool:
    if not color_guess:
        return False

    parts = _split_color_guess(color_guess)
    if not parts:
        return False

    allowed: Set[str] = set()
    for p in parts:
        allowed |= _color_aliases_one(p)

    c1 = str(it.get("COLOR_CLASS1") or "").strip()
    c2 = str(it.get("COLOR_CLASS2") or "").strip()
    return (c1 in allowed) or (c2 in allowed)


def _shape_aliases(shape: Optional[str]) -> Set[str]:
    if not shape:
        return set()
    s = shape.strip()
    if s == "타원형":
        return {"타원형", "장방형"}
    if s == "장방형":
        return {"장방형", "타원형"}
    return {s}


def _item_has_shape(it: dict, shape_guess: Optional[str]) -> bool:
    if not shape_guess:
        return False
    shapes = _shape_aliases(shape_guess)
    s = str(it.get("DRUG_SHAPE") or "").strip()
    return s in shapes


def score_candidate(
    it: dict,
    imprint: str,
    color_guess: Optional[str],
    shape_guess: Optional[str],
    is_capsule: bool = False,
) -> int:
    target = _norm(imprint)
    if not target:
        return 0

    p_front = _norm(it.get("PRINT_FRONT"))
    p_back = _norm(it.get("PRINT_BACK"))
    p_toks = _tokens_set(p_front) | _tokens_set(p_back)

    score = 0

    # 1) 각인 토큰 매칭 (핵심)
    for t in _tokens(target):
        variants = _tok_variants(t)
        if variants & p_toks:
            score += _tok_weight(t)

    # 2) shape/color는 보조 가점만 (비전 오판에 끌려가지 않도록 약하게)
    if _item_has_shape(it, shape_guess):
        score += 8
    if _item_has_color(it, color_guess):
        score += 8

    # 3) 캡슐이면 약간 가점 (특히 타원형/장방형에서 도움)
    if is_capsule:
        score += 15

    return score
