import os, sys
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.imprint_ai import ImprintCharDetector, decode_detections_to_string
from app.services.cropper import crop_and_segment_pill
from app.services.pill_features import tight_crop_to_mask
from app.services.imprint_roi import extract_imprint_roi
from app.services.imprint_ai import make_ring_mask, decode_detections_to_string

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
crop, mask = extract_imprint_roi(crop, mask)


h, w = crop.shape[:2]

# 1) 좌측(흰색) 반쪽: D-W, PAC 잘 나오는 편
x0, x1 = int(w * 0.00), int(w * 0.55)
y0, y1 = int(h * 0.10), int(h * 0.90)

crop = crop[y0:y1, x0:x1]
mask = mask[y0:y1, x0:x1]

# 원형일 때(캡슐 아님)만 링 마스크 사용
ring = make_ring_mask(mask, inner_px=28, outer_px=2)
dets = detector.infer(crop, mask=ring)

h, w = crop.shape[:2]
decoded = decode_detections_to_string(dets, img_w=w, img_h=h, radial=True)


for d in dets:
    print(d)

#
# print("decoded =", text)
# print("num dets =", len(dets))

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
