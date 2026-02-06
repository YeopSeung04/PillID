# app/main.py
print("### RUNNING app/main.py v-MULTI-ROI-STABLE ###")

import re
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.schemas import IdentifyResponse, Candidate, PillResult
from app.utils.image_io import read_upload_to_bgr
from app.utils.debug_runs import make_run_dir
from app.services.mfds_cache import mfds_cache
from app.services.vision import guess_shape_color_multi
from app.services.ocr import extract_imprint_text_roi
from app.services.ranker import score_candidate, match_level


MFDS_PAGES = 50
MFDS_ROWS = 100
MFDS_TTL_SECONDS = 6 * 60 * 60

app = FastAPI(title="PillID MVP (MFDS + Vision + OCR + Ranker)")


def _post_fix_imprint(imprint: str) -> str:
    s = (imprint or "").upper()
    s = re.sub(r"[A-Z]?PAC[A-Z]?", "PAC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _gen_ocr_variants(imprint: str) -> list[str]:
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
        if a in s.replace("-", ""):
            out.add(s.replace("-", "").replace(a, b))

    ret = sorted(out, key=lambda x: (len(x), x))
    return ret[:12]


def _shape_aliases(shape: str | None) -> set[str]:
    if not shape:
        return set()
    if shape == "타원형":
        return {"타원형", "장방형"}
    if shape == "장방형":
        return {"장방형", "타원형"}
    return {shape}


def _color_aliases(color: str | None) -> set[str]:
    if not color:
        return set()

    parts = re.split(r"[+,/ ]+", color)
    out = set()

    for c in parts:
        if c == "연두":
            out.update({"연두", "녹색", "초록", "초록색"})
        elif c == "하양":
            out.update({"하양", "흰색", "백색"})
        elif c == "노랑":
            out.update({"노랑", "황색"})
        elif c == "파랑":
            out.update({"파랑", "청색"})
        else:
            out.add(c)

    return out


def _prefilter_items(items, shape_guess, color_guess):
    shapes = _shape_aliases(shape_guess)
    colors = _color_aliases(color_guess)

    def ok_shape(it):
        return not shapes or str(it.get("DRUG_SHAPE", "")).strip() in shapes

    def ok_color(it):
        if not colors:
            return True
        return (
            str(it.get("COLOR_CLASS1", "")).strip() in colors or
            str(it.get("COLOR_CLASS2", "")).strip() in colors
        )

    strong = [it for it in items if ok_shape(it) and ok_color(it)]
    if len(strong) >= 30:
        return strong, "strong"

    mid = [it for it in items if ok_shape(it)]
    if len(mid) >= 30:
        return mid, "mid"

    weak = [it for it in items if ok_color(it)]
    if len(weak) >= 30:
        return weak, "weak"

    return items, "none"


@app.on_event("startup")
async def on_startup():
    async def _load():
        await mfds_cache.refresh(pages=MFDS_PAGES, rows=MFDS_ROWS)
        print(f"[MFDS] cache loaded: {len(mfds_cache.items)} items")

    asyncio.create_task(_load())


@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only jpg/png/webp allowed")

    run_dir = make_run_dir(base_dir="debug_runs")
    bgr = await read_upload_to_bgr(file, save_path=f"{run_dir}/input_original.jpg")

    if not mfds_cache.is_ready():
        return IdentifyResponse(note="MFDS 캐시 로딩 중입니다.")

    rois_info = guess_shape_color_multi(bgr, debug_dir=run_dir)
    pills: list[PillResult] = []

    for idx, info in enumerate(rois_info, start=1):
        x, y, w, h = info["roi"]
        roi_img = info.get("roi_img")
        if roi_img is None:
            roi_img = bgr[y:y + h, x:x + w]

        shape_guess = info.get("shape")
        color_guess = info.get("color")
        is_capsule = bool(info.get("is_capsule", False))

        imprint_raw = extract_imprint_text_roi(
            roi_img,
            debug_dir=run_dir,
            debug_prefix=f"pill{idx}",
        )
        imprint = _post_fix_imprint(imprint_raw)
        ocr_variants = _gen_ocr_variants(imprint)

        if not imprint:
            pills.append(
                PillResult(
                    roi_index=idx,
                    roi=(x, y, w, h),
                    ocr_text="",
                    ocr_variants=[],
                    color_guess=color_guess,
                    shape_guess=shape_guess,
                    is_capsule=is_capsule,
                    candidates=[],
                    note="OCR 실패",
                )
            )
            continue

        toks = set()
        for s in [imprint] + ocr_variants:
            toks.update(_tokens(s))

        candidate_idx = set()
        for t in toks:
            candidate_idx.update(mfds_cache.print_index.get(t, []))

        items_seed = (
            [mfds_cache.items[i] for i in candidate_idx]
            if candidate_idx else mfds_cache.items
        )

        filtered, mode = _prefilter_items(items_seed, shape_guess, color_guess)

        scored = []
        for it in filtered:
            s = score_candidate(it, imprint, color_guess, shape_guess, is_capsule=is_capsule)
            if s > 0:
                scored.append((s, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:5]

        candidates = [
            Candidate(
                item_name=str(it.get("ITEM_NAME")),
                entp_name=str(it.get("ENTP_NAME")) if it.get("ENTP_NAME") else None,
                score=int(s),
                match_level=match_level(int(s)),
                imprint=str(it.get("PRINT_FRONT") or it.get("PRINT_BACK")),
                color=str(it.get("COLOR_CLASS1")),
                shape=str(it.get("DRUG_SHAPE")),
            )
            for s, it in top
        ]

        pills.append(
            PillResult(
                roi_index=idx,
                roi=(x, y, w, h),
                ocr_text=imprint,
                ocr_variants=ocr_variants,
                color_guess=color_guess,
                shape_guess=shape_guess,
                is_capsule=is_capsule,
                candidates=candidates,
                note=f"prefilter={mode}",
            )
        )

    return IdentifyResponse(
        pills=pills,
        note="본 결과는 참고용이며 복용 전 약사/의사 상담이 필요합니다.",
    )
