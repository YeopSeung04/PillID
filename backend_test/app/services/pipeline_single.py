from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import cv2

from app.services.cropper import crop_and_segment_pill
from app.services.pill_features import (
    guess_color_from_mask,
    guess_shape_from_mask,
    is_capsule_from_mask,
    tight_crop_to_mask,
)
from app.services.imprint_ai import (
    ImprintCharDetector,
    decode_detections_to_string,
    make_ring_mask_ratio,
    _prep_deboss_for_yolo,
)
from app.services.ocr import (
    extract_imprint_text_roi,
    extract_imprint_text_round_unwrap_ocr,
)
from app.services.mfds_cache import mfds_cache
from app.services.ranker import score_candidate, match_level


_TOK_RE = re.compile(r"[A-Z0-9\-]{2,}")


def _tokens(s: str) -> List[str]:
    return _TOK_RE.findall((s or "").upper())


# app/services/pipeline_single.py (상단)
def classify_imprint_type(crop_bgr, mask):
    import numpy as np, cv2

    m = (mask > 0).astype(np.uint8)
    b, g, r = cv2.split(crop_bgr)
    rg = cv2.absdiff(r, g)
    ink_score = float((rg[m > 0].mean() if m.any() else rg.mean()) / 255.0)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    deboss_score = float((mag[m > 0].mean() if m.any() else mag.mean()) / 255.0)

    if ink_score > 0.08:
        return "printed"
    if deboss_score > 0.20:
        return "debossed"
    return "unknown"


def _post_fix_imprint(imprint: str) -> str:
    s = (imprint or "").upper()
    s = re.sub(r"[^A-Z0-9\-\s\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("D-W1", "D-W").replace("DW1", "DW")
    s = re.sub(r"\b[A-Z]?PAC[A-Z]?\b", "PAC", s)

    toks = _TOK_RE.findall(s)
    seen = set()
    out = []
    for t in toks:
        t0 = t.replace("-", "")
        if t0 in seen:
            continue
        seen.add(t0)
        out.append(t)

    out2 = []
    for t in out:
        if t.replace("-", "") == "DW":
            out2.append("D-W")
        else:
            out2.append(t)

    has_dw = any(t.replace("-", "") == "DW" for t in out2)
    has_pac = any(t == "PAC" for t in out2)

    if has_dw and has_pac:
        return "D-W PAC"
    if has_dw:
        return "D-W"
    if has_pac:
        return "PAC"
    return " ".join(out2).strip()


def _gen_ocr_variants(imprint: str) -> List[str]:
    s = (imprint or "").upper().strip()
    if not s:
        return []

    swaps = [
        ("0", "O"),
        ("O", "0"),
        ("1", "I"),
        ("I", "1"),
        ("5", "S"),
        ("S", "5"),
        ("2", "Z"),
        ("Z", "2"),
        ("8", "B"),
        ("B", "8"),
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
    bgr,
    debug_dir: Optional[str] = None,
    debug_prefix: str = "one",
    detector: Optional[ImprintCharDetector] = None,
) -> Dict:
    """
    단일 알약 파이프라인.
    - 캡슐(인쇄형): ROI OCR 우선
    - 정제(원형/음각 가능): (1) deboss-prep YOLO(radial) 시도 -> (2) unwrap OCR -> (3) ROI OCR fallback
    """
    # 1) crop + segment
    crop, mask, rgba = crop_and_segment_pill(bgr)
    crop, mask = tight_crop_to_mask(crop, mask)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_crop.jpg"), crop)
        if mask is not None and mask.size:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mask.png"), mask)
        if rgba is not None and rgba.size:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_rgba.png"), rgba)

    # 2) color/shape
    color_guess = guess_color_from_mask(crop, mask)
    shape_guess = guess_shape_from_mask(mask)
    is_capsule = is_capsule_from_mask(mask, shape_guess)

    # 3) imprint
    if detector is None:
        detector = ImprintCharDetector()

    print("[pipe] is_capsule =", is_capsule, "detector_ready =", detector.is_ready())
    imprint = ""

    if detector.is_ready():
        imprint_type = classify_imprint_type(crop, mask)
        print("[pipe] imprint_type =", imprint_type)

        if is_capsule or imprint_type == "printed":
            # 인쇄형 / 캡슐
            imprint = extract_imprint_text_roi(
                crop,
                debug_dir=debug_dir,
                debug_prefix=f"{debug_prefix}_ocr" if debug_dir else None,
            )

        elif imprint_type == "debossed":
            # 음각형 (ROPECIA 같은 놈들)
            ring = make_ring_mask_ratio(mask, inner_ratio=0.45, outer_ratio=0.01)
            prep = _prep_deboss_for_yolo(crop)
            dets = detector.infer(prep, mask=ring, apply_preproc=False)
            print("[pipe] dets(deboss) =", [(d.ch, round(d.conf, 3)) for d in dets[:8]])

            h, w = crop.shape[:2]
            imprint = decode_detections_to_string(dets, img_w=w, img_h=h, radial=True)

            # YOLO 실패 시 unwrap OCR
            if not imprint or len(imprint.replace(" ", "")) < 3:
                imprint = extract_imprint_text_round_unwrap_ocr(
                    crop,
                    pill_mask=mask,
                    debug_dir=debug_dir,
                    debug_prefix=f"{debug_prefix}_unwrap" if debug_dir else None,
                )

        else:
            # unknown → 안전하게 OCR
            imprint = extract_imprint_text_roi(
                crop,
                debug_dir=debug_dir,
                debug_prefix=f"{debug_prefix}_ocr" if debug_dir else None,
            )

    print("[pipe] imprint(decoded) =", imprint)

    # 최후 fallback: ROI OCR
    if not imprint:
        imprint = extract_imprint_text_roi(
            crop,
            debug_dir=debug_dir,
            debug_prefix=f"{debug_prefix}_ocr" if debug_dir else None,
        )

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

    # 5) scoring
    scored = []
    for it in seed_items:
        s = score_candidate(
            it,
            imprint,
            color_guess=color_guess,
            shape_guess=shape_guess,
            is_capsule=is_capsule,
        )
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
