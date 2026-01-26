# app/schemas.py
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel


class Candidate(BaseModel):
    item_name: str
    entp_name: Optional[str] = None
    score: int
    match_level: str
    imprint: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None


class IdentifyResponse(BaseModel):
    ocr_text: str
    ocr_variants: List[str]
    color_guess: Optional[str] = None
    shape_guess: Optional[str] = None
    candidates: List[Candidate]
    note: str
