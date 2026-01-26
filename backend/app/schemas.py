# app/schemas.py
from __future__ import annotations

from typing import Optional, List, Tuple
from pydantic import BaseModel


class Candidate(BaseModel):
    item_name: str
    entp_name: Optional[str] = None
    score: int
    match_level: str
    imprint: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None


class PillResult(BaseModel):
    roi_index: int
    roi: Tuple[int, int, int, int]  # (x,y,w,h)
    ocr_text: str
    ocr_variants: List[str] = []
    color_guess: Optional[str] = None     # "빨강+파랑"
    shape_guess: Optional[str] = None
    is_capsule: Optional[bool] = None
    candidates: List[Candidate] = []
    note: str = ""


class IdentifyResponse(BaseModel):
    # ✅ 멀티 결과
    pills: List[PillResult] = []
    note: str = ""

    # ✅ 하위호환(기존 단일 응답도 유지)
    ocr_text: str = ""
    ocr_variants: List[str] = []
    color_guess: Optional[str] = None
    shape_guess: Optional[str] = None
    candidates: List[Candidate] = []
