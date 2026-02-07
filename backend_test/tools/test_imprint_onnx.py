import os
import cv2

from app.services.imprint_ai import ImprintCharDetector, decode_detections_to_string
from app.services.cropper import crop_and_segment_pill
from app.services.pill_features import tight_crop_to_mask

# 사용법:
#   python tools/test_imprint_onnx.py path/to/image.jpg
import sys

if len(sys.argv) < 2:
    print("usage: python tools/test_imprint_onnx.py <image_path>")
    raise SystemExit(1)

img_path = sys.argv[1]
bgr = cv2.imread(img_path)
if bgr is None:
    raise RuntimeError(f"failed to read: {img_path}")

onnx = os.getenv("IMPRINT_YOLO_ONNX")
print("IMPRINT_YOLO_ONNX =", onnx)

detector = ImprintCharDetector()
print("detector ready =", detector.is_ready())

crop, mask, rgba = crop_and_segment_pill(bgr)
crop, mask = tight_crop_to_mask(crop, mask)

dets = detector.infer(crop, mask=mask)
text = decode_detections_to_string(dets)

print("decoded =", text)
print("num dets =", len(dets))

# 결과 시각화 저장
vis = crop.copy()
for d in dets:
    x0, y0, x1, y1 = d.xyxy
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(vis, f"{d.ch}:{d.conf:.2f}", (x0, max(15, y0-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("tools/_test_crop.jpg", crop)
cv2.imwrite("tools/_test_mask.png", mask)
cv2.imwrite("tools/_test_vis.jpg", vis)
print("saved: tools/_test_crop.jpg, tools/_test_mask.png, tools/_test_vis.jpg")
