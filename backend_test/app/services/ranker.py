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
    s = (s or "").upper().strip()
    # 흔한 노이즈: DW1 / D-W1 -> DW / D-W 취급
    s = s.replace("D-W1", "D-W").replace("DW1", "DW")
    return s



def _tokens_list(s: str) -> list[str]:
    """원본 문자열에서 토큰 리스트(순서 유지)가 필요할 때."""
    return _TOKEN_RE.findall(_norm(s))


def _tokens_set(s: str) -> Set[str]:
    """토큰 집합(매칭용)."""
    return set(_tokens_list(s))


def _tok_variants(t: str) -> Set[str]:
    """
    OCR/YOLO 토큰 변형 흡수 (실전용):
    - 하이픈 제거: D-W <-> DW
    - 자주 헷갈리는 문자 치환:
        VV <-> W
        S  <-> C
        0  <-> O
        1  <-> I
    - 조합(1~2회)까지 생성해서 recall을 올림
    """
    t = _norm(t)
    if not t:
        return set()

    # 0) 기본 후보
    base = {t, t.replace("-", "")}

    # 1) 치환 규칙 (단방향/양방향 섞음)
    subs = [
        ("VV", "W"),   # YOLO가 W를 VV로 자주 뽑음
        ("W", "VV"),   # 반대도 허용
        ("S", "C"),    # C/S 혼동
        ("C", "S"),
        ("0", "O"),    # 0/O 혼동
        ("O", "0"),
        ("1", "I"),    # 1/I 혼동
        ("I", "1"),
    ]

    def apply_once(s: str) -> Set[str]:
        out = set()
        for a, b in subs:
            if a in s:
                out.add(s.replace(a, b))
        return out

    # 2) 1회 치환
    v1 = set()
    for s in list(base):
        v1 |= apply_once(s)

    # 3) 2회 치환(폭발 방지 위해 1회 결과에 한 번만 더)
    v2 = set()
    for s in list(v1):
        v2 |= apply_once(s)

    # 4) 최종 집합 (+ 하이픈 제거를 다시 한번 적용)
    out = set()
    for s in (base | v1 | v2):
        out.add(s)
        out.add(s.replace("-", ""))

    # 너무 짧은 건 버림(오탐 방지)
    out = {s for s in out if len(s) >= 2}

    return out


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
    if t0 in ("D-VV", "DVV") or t0_no == "DVV":
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

    # 입력은 "연두+하양" 같이 복합일 수 있음
    parts = re.split(r"[+,/\s]+", color.strip())
    parts = [p for p in parts if p]
    if not parts:
        return set()

    out: Set[str] = set()
    for c in parts:
        # 표기 흔들림 흡수
        if c == "주황":
            out.update({"주황", "주황색", "연주황", "적주황"})
        elif c == "빨강":
            out.update({"빨강", "적색", "빨간", "적"})
        elif c == "하양":
            out.update({"하양", "흰색", "백색"})
        elif c == "파랑":
            out.update({"파랑", "청색", "파란"})
        elif c == "노랑":
            out.update({"노랑", "황색", "노란"})
        elif c == "연두":
            out.update({"연두", "녹색", "초록", "초록색"})
        else:
            out.add(c)

    return out


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
    **kwargs,
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

    # ✅ imprint가 비어도 후보가 '0명'이 되면 UX가 망가짐.
    #    색/모양만으로도 약하게라도 점수를 줘서 Top-N을 열어둔다.
    if not target:
        s = 0
        has_shape = _item_has_shape(it, shape_guess)
        has_color = _item_has_color(it, color_guess)
        if has_shape:
            s += 25
        if has_color:
            s += 25
        if has_shape and has_color:
            s += 20  # 둘 다 맞으면 보너스
        return s

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
    toks = list(dict.fromkeys(_tokens_list(target)))  # 순서 유지 중복 제거

    matched = set()

    for t in toks:
        variants = _tok_variants(t)
        if not variants:
            continue
        if variants & p_toks:
            score += _tok_weight(t)
            matched.add(t.replace("-", ""))  # 정규화해서 기록

    # PAC만 맞은 경우 HIGH로 못 가게 상한
    matched_no = {m.replace("-", "") for m in matched}
    if matched_no == {"PAC"}:
        score = min(score, 69)  # HIGH(70) 못 넘게
    # 2개 이상 토큰이 맞으면 보너스 (DW+PAC 같은 조합을 확실히 1등으로)
    if len(matched_no) >= 2:
        score += 25

    # 색/모양 가점 (너무 강하게 하면 비전 오판에 끌려감)
    if _item_has_shape(it, shape_guess):
        score += 8
    if _item_has_color(it, color_guess):
        score += 8

    return score
