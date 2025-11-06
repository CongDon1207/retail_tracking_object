# main.py
from track.tracker_factory import create_tracker
import cv2

if __name__ == "__main__":
    # --- cấu hình ---
    model_name = "yolo11l.pt"   # chỉ cần tên file, tự tìm trong detect/models/
    video_path = "data/video2.mp4"
    tracker_type = "botsort"    # hoặc "bytetrack"

    # chỉ track người
    class_filter = [0] 

    # --- khởi tạo tracker ---
    tracker = create_tracker(tracker_type, model_name)

    # --- chạy tracking ---
    for record in tracker.track(video_path, show=False, classes=class_filter):
        frame = record["frame"]  # lấy frame từ record
        for obj in record["objects"]:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f"{obj['label']} ID:{obj['id']} {obj['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("YOLO11 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()



