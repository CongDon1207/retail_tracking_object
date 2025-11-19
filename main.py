# main.py
from track.tracker_factory import create_tracker
from emit.json_emitter import JsonEmitter
from datetime import datetime, timezone
import uuid
import cv2

if __name__ == "__main__":
    # --- cấu hình ---
    model_name = "yolo11l.pt"   # chỉ cần tên file, tự tìm trong detect/models/
    video_path = "data/video3.mp4"
    tracker_type = "botsort"    # hoặc "bytetrack"
    class_filter = [0]          # chỉ track người
    out_jsonl = "metadata/video.jsonl"

    # --- metadata chung ---
    pipeline_run_id = uuid.uuid4().hex
    source = {"store_id": "store_01", "camera_id": "cam_01", "stream_id": "stream_01"}

    # --- khởi tạo ---
    tracker = create_tracker(tracker_type, model_name)
    emitter = JsonEmitter(out_jsonl)

    try:
        # --- chạy tracking ---
        for idx, record in enumerate(tracker.track(video_path, show=False, classes=class_filter), start=1):
            frame = record["frame"]  # lấy frame từ record
            H, W = frame.shape[:2]

            # vẽ bbox và label với background
            for obj in record["objects"]:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                label = f"{obj['label']} ID:{obj['id']} {obj['conf']:.2f}"
                
                # Vẽ bbox màu lightblue
                color = (255, 191, 0)  # BGR: lightblue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Tính kích thước text để vẽ background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45  # Giảm từ 0.6 xuống 0.45
                thickness = 1      # Giảm từ 2 xuống 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Vẽ background cho text (màu tối hơn)
                cv2.rectangle(frame, 
                             (x1, y1 - text_h - baseline - 5),
                             (x1 + text_w, y1),
                             color, -1)  # Fill
                
                # Vẽ text màu đen trên background
                cv2.putText(frame, label, (x1, y1 - baseline - 2),
                           font, font_scale, (0, 0, 0), thickness)

            # map sang detections theo schema đã thống nhất
            detections = []
            for j, obj in enumerate(record["objects"]):
                x1, y1, x2, y2 = map(float, obj["bbox"])
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w/2.0
                cy = y1 + h/2.0
                detections.append({
                    "det_id": f"{idx}-{j}",
                    "class": obj.get("label", "person"),
                    "class_id": int(obj.get("cls", 0)),
                    "conf": float(obj.get("conf", 0.0)),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_norm": {"x": x1/W, "y": y1/H, "w": w/W, "h": h/H},
                    "centroid": {"x": int(cx), "y": int(cy)},
                    "centroid_norm": {"x": cx/W, "y": cy/H},
                    "track_id": None if obj.get("id", -1) < 0 else int(obj["id"]),
                })

            # emit JSON (1 dòng / frame)
            emitter.emit_frame(
                pipeline_run_id=pipeline_run_id,
                source=source,
                frame_index=idx,
                capture_ts_iso=datetime.now(timezone.utc).isoformat(),
                image_size={"width": W, "height": H},
                detections=detections,
                runtime={
                    "model_name": model_name,
                    "tracker_type": tracker_type,
                    "conf_thres": 0.25,
                    "class_filter": class_filter
                },
                source_uri=video_path
            )

            # hiển thị
            cv2.imshow("YOLO11 Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        emitter.close()
        cv2.destroyAllWindows()
