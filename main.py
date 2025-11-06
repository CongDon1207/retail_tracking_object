from detect.yolo_detector import YoloDetector
from ingest.CVSource import ingest_video
import cv2

detector = YoloDetector(conf_thres=0.25)

# chỉ lấy lớp "person" (COCO id = 0)
class_filter = [0]

for item in ingest_video("data/video2.mp4", realtime=True):
    frame = item["frame"]
    detections = detector.predict(frame, class_filter=class_filter)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["label"]} {det["conf"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO11 Detect", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
