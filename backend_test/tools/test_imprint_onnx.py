# tools/test_imprint_onnx.py
import os
import sys
import asyncio
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app.services.pipeline_single import identify_one_pill
from app.services.imprint_ai import ImprintCharDetector
from app.services.mfds_cache import mfds_cache


async def _load_mfds():
    await mfds_cache.ensure_loaded(pages=50, rows=100, ttl_seconds=30 * 24 * 3600)


def main():
    if len(sys.argv) < 2:
        print("usage: python tools/test_imprint_onnx.py <image_path>")
        raise SystemExit(1)

    img_path = sys.argv[1]
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"failed to read: {img_path}")

    asyncio.run(_load_mfds())

    print("IMPRINT_YOLO_ONNX =", os.getenv("IMPRINT_YOLO_ONNX"))
    print("IMPRINT_DATA_YAML =", os.getenv("IMPRINT_DATA_YAML"))

    detector = ImprintCharDetector()
    print("detector ready =", detector.is_ready())

    out = identify_one_pill(
        bgr,
        debug_dir="tools",
        debug_prefix="one",
        detector=detector,
    )

    print("imprint =", out.get("imprint"))
    print("color_guess =", out.get("color_guess"))
    print("shape_guess =", out.get("shape_guess"))
    print("is_capsule =", out.get("is_capsule"))
    print("top_candidates =", out.get("candidates", [])[:3])
    print("saved debug under tools/ (one_crop.jpg, one_mask.png, one_ring.png, one_deboss_prep.jpg, one_unwrap_*.png etc)")


if __name__ == "__main__":
    main()
