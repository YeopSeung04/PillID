# app/main.py
print("### RUNNING app/main.py v-MULTI-ROI-SEED-PREFILTER-RANK ###")

import re
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.schemas import IdentifyResponse, Candidate, PillResult
from app.utils.image_io import read_upload_to_bgr
from app.utils.debug_runs import make_run_dir
from app.services.mfds_cache import mfds_cache
from app.services.vision import guess_shape_color_multi
from app.services.ocr import extract_imprint_text_roi
from app.services.ranker import score_candidate, match_level, _tokens  # ✅ 여기 걸 씀


MFDS_PAGES = 50
MFDS_ROWS = 100
MFDS_TTL_SECONDS = 6 * 60 * 60

app = FastAPI(title="PillID MVP (Cached MFDS + OCR + Vision + DebugRuns)")


def _post_fix_imprint(imprint: str) -> str:
    s = (imprint or "").upper()

    # PAC 주변 꼬리 제거(붙여읽기 완화)
    s = re.sub(r"[A-Z]?PAC[A-Z]?", "PAC", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _shape_aliases(shape: str | None) -> set[str]:
    if not shape:
        return set()
    shape = shape.strip()
    if shape == "타원형":
        return {"타원형", "장방형"}
    if shape == "장방형":
        return {"장방형", "타원형"}
    return {shape}


def _color_aliases(color: str | None) -> set[str]:
    if not color:
        return set()

    parts = re.split(r"[+,/ ]+", color.strip())
    out: set[str] = set()

    def add_one(c: str):
        c = c.strip()
        if not c:
            return
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

    for p in parts:
        add_one(p)

    return out


def _prefilter_items(items_all, shape_guess, color_guess):
    shapes = _shape_aliases(shape_guess)
    colors = _color_aliases(color_guess)

    def shape_ok(it):
        if not shapes:
            return True
        v = it.get("DRUG_SHAPE")
        return (v is not None) and (str(v).strip() in shapes)

    def color_ok(it):
        if not colors:
            return True
        c1 = it.get("COLOR_CLASS1")
        c2 = it.get("COLOR_CLASS2")

        s1 = str(c1).strip() if c1 else ""
        s2 = str(c2).strip() if c2 else ""

        # ✅ OR 조건
        return (s1 in colors) or (s2 in colors)

    strong = [it for it in items_all if shape_ok(it) and color_ok(it)]
    if len(strong) >= 30:
        return strong, "strong"

    mid = [it for it in items_all if shape_ok(it)]
    if len(mid) >= 30:
        return mid, "mid"

    weak = [it for it in items_all if color_ok(it)]
    if len(weak) >= 30:
        return weak, "weak"

    return items_all, "none"


@app.on_event("startup")
async def on_startup():
    async def _load():
        try:
            await mfds_cache.refresh(pages=MFDS_PAGES, rows=MFDS_ROWS)
            print(f"[MFDS] cache loaded: {len(mfds_cache.items)} items", flush=True)
        except Exception as e:
            print(f"[MFDS] cache preload failed: {e}", flush=True)

    asyncio.create_task(_load())


@app.get("/health")
async def health():
    return {
        "ok": True,
        "mfds_cache_ready": mfds_cache.is_ready(),
        "mfds_cache_loading": mfds_cache.loading,
        "mfds_items": len(mfds_cache.items),
        "mfds_loaded_at": mfds_cache.loaded_at,
        "ttl_seconds": MFDS_TTL_SECONDS,
        "mfds_progress": getattr(mfds_cache, "progress", None),
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only jpg/png/webp allowed")

    run_dir = make_run_dir(base_dir="debug_runs")
    print(f"[DEBUG] run_dir = {run_dir}", flush=True)

    try:
        bgr = await read_upload_to_bgr(file, save_path=f"{run_dir}/input_original.jpg")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 캐시 준비 확인
    if not mfds_cache.is_ready():
        return IdentifyResponse(
            pills=[],
            note="MFDS 캐시를 로딩 중입니다. /health에서 mfds_items가 채워진 뒤 다시 시도하세요.",
        )

    # TTL stale 갱신 (요청은 기존 캐시로 처리)
    if mfds_cache.is_stale(MFDS_TTL_SECONDS) and not mfds_cache.loading:
        asyncio.create_task(mfds_cache.refresh(pages=MFDS_PAGES, rows=MFDS_ROWS))

    # -------------------------
    # 1) 멀티 ROI 추정
    # -------------------------
    rois_info = guess_shape_color_multi(
        bgr,
        debug_dir=run_dir,
        debug_prefix="vision",
    )

    # ROI가 하나도 안 잡히면 fallback
    if not rois_info:
        rois_info = [
            {"roi": (0, 0, bgr.shape[1], bgr.shape[0]), "shape": None, "color": None, "colors": [], "is_capsule": False}
        ]

    pills: list[PillResult] = []

    # -------------------------
    # 2) ROI별 식별
    # -------------------------
    for idx, info in enumerate(rois_info, start=1):
        x, y, w, h = info["roi"]
        roi_img = bgr[y:y + h, x:x + w]

        shape_guess = info.get("shape")
        color_guess = info.get("color")     # "빨강+파랑"
        is_capsule = bool(info.get("is_capsule", False))

        imprint_raw = extract_imprint_text_roi(
            roi_img,
            debug_dir=run_dir,
            debug_prefix=f"pill{idx}",
        )
        imprint = _post_fix_imprint(imprint_raw)

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

        toks = _tokens(imprint)

        # 2-1) seed from print_index (하이픈 제거 variant 포함)
        candidate_idx = set()
        for t in toks:
            variants = {t, t.replace("-", "")}
            for v in variants:
                for i in mfds_cache.print_index.get(v, []):
                    candidate_idx.add(i)

        items_seed = [mfds_cache.items[i] for i in candidate_idx] if candidate_idx else mfds_cache.items

        # 2-2) prefilter
        filtered, mode = _prefilter_items(items_seed, shape_guess, color_guess)

        print(
            f"[DEBUG] roi={idx} seed={len(candidate_idx) if candidate_idx else 'ALL'} | "
            f"prefilter({mode}): seed={len(items_seed)} -> filtered={len(filtered)} | "
            f"shape={shape_guess} color={color_guess} is_capsule={is_capsule} toks={toks}",
            flush=True
        )

        # 2-3) rank
        scored = []
        for it in filtered:
            s = score_candidate(
                it,
                imprint,
                color_guess,
                shape_guess,
                is_capsule=is_capsule,   # ✅ 여기
            )
            if s > 0:
                scored.append((s, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:5]

        candidates = [
            Candidate(
                item_name=str(it.get("ITEM_NAME") or "UNKNOWN"),
                entp_name=str(it.get("ENTP_NAME")) if it.get("ENTP_NAME") else None,
                score=int(s),
                match_level=match_level(int(s)),
                imprint=str(it.get("PRINT_FRONT") or it.get("PRINT_BACK"))
                if (it.get("PRINT_FRONT") or it.get("PRINT_BACK"))
                else None,
                color=str(it.get("COLOR_CLASS1")) if it.get("COLOR_CLASS1") else None,
                shape=str(it.get("DRUG_SHAPE")) if it.get("DRUG_SHAPE") else None,
            )
            for s, it in top
        ]

        pills.append(
            PillResult(
                roi_index=idx,
                roi=(x, y, w, h),
                ocr_text=imprint,
                ocr_variants=[],
                color_guess=color_guess,
                shape_guess=shape_guess,
                is_capsule=is_capsule,
                candidates=candidates,
                note=f"prefilter={mode}, seed={len(candidate_idx) if candidate_idx else 'ALL'}",
            )
        )

    # -------------------------
    # 3) best 1개 선정 (하위호환)
    # -------------------------
    def _best_score(p: PillResult) -> int:
        return int(p.candidates[0].score) if p.candidates else -1

    best = max(pills, key=_best_score) if pills else None

    if not best or not best.candidates:
        return IdentifyResponse(
            pills=pills,
            note="각인은 인식했지만 매칭 후보를 찾지 못했습니다. 각인을 더 크게/선명하게 찍어 다시 시도하세요.",
        )

    return IdentifyResponse(
        pills=pills,
        ocr_text=best.ocr_text,
        ocr_variants=best.ocr_variants,
        color_guess=best.color_guess,
        shape_guess=best.shape_guess,
        candidates=best.candidates,
        note="본 결과는 의약품 식별 참고용 후보이며 진단/복용판단이 아닙니다. 복용 전 약사/의사와 상담하세요.",
    )
