import cv2
from ingest.CVSource import ingest_video
import torch 

for item in ingest_video("data/video.mp4"):
    cv2.imshow("frame", item["frame"])
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()

