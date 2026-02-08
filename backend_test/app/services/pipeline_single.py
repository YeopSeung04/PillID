# app/services/pipeline_single.py
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.cropper import crop_and_segment_pill
from app.services.pill_features import (
    guess_color_from_mask,
    guess_shape_from_mask,
    is_capsule_from_mask,
    tight_crop_to_mask,
)
from app.services.imprint_ai import ImprintCharDetector, decode_detections_to_string
from app.services.ocr import extract_imprint_text_roi
from app.services.mfds_cache import mfds_cache
from app.services.ranker import score_candidate, match_level
from app.services.imprint_ai import ImprintCharDetector, decode_detections_to_string, make_ring_mask


_TOK_RE = re.compile(r"[A-Z0-9\-]{2,}")


def _tokens(s: str) -> List[str]:
    return _TOK_RE.findall((s or "").upper())


def _post_fix_imprint(imprint: str) -> str:
    s = (imprint or "").upper()
    s = re.sub(r"[A-Z]?PAC[A-Z]?", "PAC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _gen_ocr_variants(imprint: str) -> List[str]:
    s = (imprint or "").upper().strip()
    if not s:
        return []

    swaps = [
        ("0", "O"), ("O", "0"),
        ("1", "I"), ("I", "1"),
        ("5", "S"), ("S", "5"),
        ("2", "Z"), ("Z", "2"),
        ("8", "B"), ("B", "8"),
    ]

    out = {s, s.replace("-", ""), s.replace(" ", "")}
    for a, b in swaps:
        if a in s:
            out.add(s.replace(a, b))
        ss = s.replace("-", "")
        if a in ss:
            out.add(ss.replace(a, b))

    ret = sorted(out, key=lambda x: (len(x), x))
    return ret[:12]


def identify_one_pill(
    bgr: np.ndarray,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "one",
    detector: Optional[ImprintCharDetector] = None,
) -> Dict:
    """단일 알약 파이프라인. (dict 반환 -> main에서 schema로 감싸기)"""

    # 1) crop + segment
    crop, mask, rgba = crop_and_segment_pill(bgr)
    crop, mask = tight_crop_to_mask(crop, mask)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_crop.jpg"), crop)
        if mask.size:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mask.png"), mask)
        if rgba.size:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_rgba.png"), rgba)

    # 2) color/shape
    color_guess = guess_color_from_mask(crop, mask)
    shape_guess = guess_shape_from_mask(mask)
    is_capsule = is_capsule_from_mask(mask, shape_guess)

    # 3) imprint (YOLO char-detector 우선, 없으면 OCR fallback)
    imprint = ""
    print("[pipe] is_capsule =", is_capsule, "detector_ready =", detector.is_ready())
    if detector is None:
        detector = ImprintCharDetector()  # env IMPRINT_YOLO_ONNX 있으면 로드

    if detector.is_ready():
        if is_capsule:
            dets = detector.infer(crop, mask=mask)
            imprint = decode_detections_to_string(dets, radial=False)
            print("[pipe] dets =", len(dets), "sample =", dets[:5])

        else:
            ring = make_ring_mask(mask, inner_px=28, outer_px=2)
            # ✅ 여기
            if debug_dir:
                cv2.imwrite(
                    os.path.join(debug_dir, f"{debug_prefix}_ring.png"),
                    ring
                )
            cv2.imwrite("tools/_ring.png", ring)

            dets = detector.infer(crop, mask=ring)
            h, w = crop.shape[:2]
            imprint = decode_detections_to_string(
                dets,
                img_w=w,
                img_h=h,
                radial=True
            )

    print("[pipe] imprint(decoded) =", imprint)

    if not imprint:
        imprint_raw = extract_imprint_text_roi(
            crop,
            debug_dir=debug_dir,
            debug_prefix=f"{debug_prefix}_ocr" if debug_dir else None,
        )
        imprint = imprint_raw

    imprint = _post_fix_imprint(imprint)
    ocr_variants = _gen_ocr_variants(imprint)

    # 4) candidate seed
    toks = set()
    for s in [imprint] + ocr_variants:
        toks.update(_tokens(s))

    candidate_idx = set()
    for t in toks:
        candidate_idx.update(mfds_cache.print_index.get(t, []))

    seed_items = [mfds_cache.items[i] for i in candidate_idx] if candidate_idx else mfds_cache.items

    # 5) scoring (imprint 없어도 ranker가 color/shape로 점수 내게 수정해둠)
    scored = []
    for it in seed_items:
        s = score_candidate(it, imprint, color_guess=color_guess, shape_guess=shape_guess, is_capsule=is_capsule)
        if s > 0:
            scored.append((int(s), it))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    candidates = []
    for s, it in top:
        candidates.append(
            {
                "item_name": str(it.get("ITEM_NAME")),
                "entp_name": str(it.get("ENTP_NAME")) if it.get("ENTP_NAME") else None,
                "score": int(s),
                "match_level": match_level(int(s)),
                "imprint": str(it.get("PRINT_FRONT") or it.get("PRINT_BACK")),
                "color": str(it.get("COLOR_CLASS1")) if it.get("COLOR_CLASS1") else None,
                "shape": str(it.get("DRUG_SHAPE")) if it.get("DRUG_SHAPE") else None,
            }
        )

    return {
        "roi": (0, 0, int(crop.shape[1]), int(crop.shape[0])),
        "crop_bgr": crop,
        "mask": mask,
        "rgba": rgba,
        "imprint": imprint,
        "ocr_variants": ocr_variants,
        "color_guess": color_guess,
        "shape_guess": shape_guess,
        "is_capsule": is_capsule,
        "candidates": candidates,
    }
