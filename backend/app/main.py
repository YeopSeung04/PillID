# app/main.py
print("### RUNNING app/main.py v-SHAPE-COLOR-PREFILTER-FULL ###")

import re
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.schemas import IdentifyResponse, Candidate
from app.utils.image_io import read_upload_to_bgr
from app.utils.debug_runs import make_run_dir
from app.services.ocr import extract_imprint_text
from app.services.vision import guess_shape_color
from app.services.ranker import score_candidate, match_level
from app.services.mfds_cache import mfds_cache

MFDS_PAGES = 50
MFDS_ROWS = 100
MFDS_TTL_SECONDS = 6 * 60 * 60

app = FastAPI(title="PillID MVP (Cached MFDS + OCR + Vision + DebugRuns)")


def _raw_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Z0-9\-]{2,}", (s or "").upper())



def _good_token(tok: str) -> bool:
    if not (2 <= len(tok) <= 8):
        return False
    if not re.search(r"[A-Z]", tok):
        return False
    return True


def _merge_tokens(*texts: str) -> str:
    seen = set()
    out = []
    for t in texts:
        for tok in _raw_tokens(t):
            if not _good_token(tok):
                continue
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
    return " ".join(out)


def _post_fix_imprint(imprint: str) -> str:
    """
    OCR 결과 노이즈를 '가볍게'만 정리.
    (과정이 너무 공격적이면 정답도 날아감)
    """
    s = (imprint or "").upper()

    # D-W 뒤에 붙는 1글자 제거
    # s = re.sub(r"(D-W)[A-Z0-9]", r"\1", s)

    # PAC 주변 꼬리 제거(붙여읽기 완화)
    s = re.sub(r"[A-Z]?PAC[A-Z]?", "PAC", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _gen_ocr_variants(imprint: str) -> list[str]:
    """Generate a small set of robust OCR variants (for seeding only)."""
    s = (imprint or "").upper().strip()
    if not s:
        return []

    # normalize spaces
    s = re.sub(r"\s+", " ", s)

    # common confusions
    swaps = [
        ("0", "O"), ("O", "0"),
        ("1", "I"), ("I", "1"),
        ("5", "S"), ("S", "5"),
        ("2", "Z"), ("Z", "2"),
        ("8", "B"), ("B", "8"),
    ]

    out: set[str] = {s, s.replace("-", ""), s.replace(" ", "")}
    for a, b in swaps:
        if a in s:
            out.add(s.replace(a, b))
        if a in s.replace("-", ""):
            out.add(s.replace("-", "").replace(a, b))

    # keep size small / deterministic
    ret = [x for x in out if x]
    ret.sort(key=lambda x: (len(x), x))
    return ret[:12]


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

    # "하양+주황" / "하양,주황" 등 분해 지원
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

def _tokens(s: str) -> list[str]:
    return re.findall(r"[A-Z0-9\-]{2,}", (s or "").upper().strip())



@app.on_event("startup")
async def on_startup():
    async def _load():
        try:
            await mfds_cache.refresh(pages=MFDS_PAGES, rows=MFDS_ROWS)
            print(f"[MFDS] cache loaded: {len(mfds_cache.items)} items")
        except Exception as e:
            print(f"[MFDS] cache preload failed: {e}")

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
        "mfds_progress": mfds_cache.progress,
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only jpg/png/webp allowed")

    run_dir = make_run_dir(base_dir="debug_runs")
    print(f"[DEBUG] run_dir = {run_dir}")

    try:
        bgr = await read_upload_to_bgr(file, save_path=f"{run_dir}/input_original.jpg")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    shape_guess, color_guess = guess_shape_color(
        bgr,
        debug_dir=run_dir,
        debug_prefix="vision",
    )
    print(f"[DEBUG] shape_guess = {shape_guess!r} color_guess = {color_guess!r}")

    h, w = bgr.shape[:2]
    left = bgr[:, : w // 2]
    right = bgr[:, w // 2 :]

    im_full = extract_imprint_text(bgr, debug_dir=run_dir, debug_prefix="full")
    im_l = extract_imprint_text(left, debug_dir=run_dir, debug_prefix="left")
    im_r = extract_imprint_text(right, debug_dir=run_dir, debug_prefix="right")

<<<<<<< Updated upstream
    imprint = _merge_tokens(im_full, im_l, im_r)
    imprint = _post_fix_imprint(imprint)
=======
    # -------------------------
    # 2) ROI별 식별
    # -------------------------
    for idx, info in enumerate(rois_info, start=1):
        x, y, w, h = info["roi"]
        roi_img_raw = bgr[y:y + h, x:x + w]
        # vision.py가 회전/정규화된 ROI를 제공하면 OCR에는 그걸 우선 사용
        roi_img = info.get("roi_img", None)
        if roi_img is None:
            roi_img = roi_img_raw
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    print("[DEBUG] im_full =", repr(im_full))
    print("[DEBUG] im_l    =", repr(im_l))
    print("[DEBUG] im_r    =", repr(im_r))
    print("[DEBUG] imprint =", repr(imprint))

<<<<<<< Updated upstream
    if not imprint:
        return IdentifyResponse(
            ocr_text="",
            ocr_variants=[],
            color_guess=color_guess,
            shape_guess=shape_guess,
            candidates=[],
            note="각인(OCR) 인식에 실패했습니다. 각인이 보이도록 정면에서 더 선명하게 촬영하세요.",
=======
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
                    ocr_variants=ocr_variants,
                    color_guess=color_guess,
                    shape_guess=shape_guess,
                    is_capsule=is_capsule,
                    candidates=[],
                    note="OCR 실패",
                )
            )
            continue

        # ✅ seed는 OCR 원문 + variants를 모두 사용
        toks = set()
        for s in [imprint] + ocr_variants:
            toks.update(_tokens(s))
        toks = sorted(toks)

        # 2-1) seed from print_index (mfds_cache에서 원형/정규화 둘 다 인덱싱됨)
        candidate_idx = set()
        for t in toks:
            for i in mfds_cache.print_index.get(t, []):
                candidate_idx.add(i)

        items_seed = [mfds_cache.items[i] for i in candidate_idx] if candidate_idx else mfds_cache.items

        # 2-2) prefilter
        filtered, mode = _prefilter_items(items_seed, shape_guess, color_guess)

        print(
            f"[DEBUG] roi={idx} seed={len(candidate_idx) if candidate_idx else 'ALL'} | "
            f"prefilter({mode}): seed={len(items_seed)} -> filtered={len(filtered)} | "
            f"shape={shape_guess} color={color_guess} is_capsule={is_capsule} toks={toks}",
            flush=True
>>>>>>> Stashed changes
        )

    if not mfds_cache.is_ready():
        return IdentifyResponse(
            ocr_text=imprint,
            ocr_variants=[],
            color_guess=color_guess,
            shape_guess=shape_guess,
            candidates=[],
            note="MFDS 캐시를 로딩 중입니다. /health에서 mfds_items가 채워진 뒤 다시 시도하세요.",
        )

    if mfds_cache.is_stale(MFDS_TTL_SECONDS) and not mfds_cache.loading:
        asyncio.create_task(mfds_cache.refresh(pages=MFDS_PAGES, rows=MFDS_ROWS))

    # 1) print_index 기반으로 1차 후보군(seed) 만들기
    toks = _tokens(imprint)  # ranker.py의 _tokens를 여기로 가져오거나 동일 구현
    candidate_idx = set()

    for t in toks:
        # 하이픈 제거 변형까지 같이 조회 (DW vs D-W)
        variants = {t, t.replace("-", "")}
        for v in variants:
            for i in mfds_cache.print_index.get(v, []):
                candidate_idx.add(i)

    if candidate_idx:
        items_seed = [mfds_cache.items[i] for i in candidate_idx]
    else:
        items_seed = mfds_cache.items

    # 2) seed에 대해 shape/color prefilter
    filtered, mode = _prefilter_items(items_seed, shape_guess, color_guess)

    print(
        f"[DEBUG] seed={len(candidate_idx) if candidate_idx else 'ALL'} | "
        f"prefilter({mode}): seed={len(items_seed)} -> filtered={len(filtered)} | "
        f"shape={shape_guess} color={color_guess} toks={toks}"
    )

    scored = []
    for it in filtered:
        s = score_candidate(it, imprint, color_guess, shape_guess)
        if s > 0:
            scored.append((s, it))

    if not scored:
        return IdentifyResponse(
            ocr_text=imprint,
            ocr_variants=[],
            color_guess=color_guess,
            shape_guess=shape_guess,
            candidates=[],
            note="각인은 인식했지만 매칭 후보를 찾지 못했습니다. 각인을 더 크게/선명하게 찍어 다시 시도하세요.",
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    candidates = []
    for s, it in top:
        candidates.append(
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
<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes
        )

    return IdentifyResponse(
        ocr_text=imprint,
        ocr_variants=[],
        color_guess=color_guess,
        shape_guess=shape_guess,
        candidates=candidates,
        note="본 결과는 의약품 식별 참고용 후보이며 진단/복용판단이 아닙니다. 복용 전 약사/의사와 상담하세요.",
    )
