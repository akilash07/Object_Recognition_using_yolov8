import cv2
import torch
from ultralytics import YOLO  # For YOLOv8; comment this if you're using YOLOv5

# For YOLOv5
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# For YOLOv8
model = YOLO('yolov8s.pt')

# IP camera URL (replace with your actual URL)
# For example, if using the IP Webcam app:
ip_camera_url = 'http://192.168.230.149:8080/video'

# Use the IP camera stream
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open IP webcam.")
    exit()

while True:
    # Capture frame-by-frame from the IP webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # YOLOv5 code
    # Convert the frame to RGB (YOLOv5 expects RGB input)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = model(frame_rgb)
    
    # YOLOv8 code
    results = model(frame)

    # YOLOv5: Get the detection results (if you're using YOLOv5, otherwise skip this for YOLOv8)
    # detections = results.xyxy[0].numpy()

    # YOLOv8: Annotate the frame with detection results (bounding boxes, labels, etc.)
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO IP Webcam Detection", annotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the IP webcam stream and close windows
cap.release()
cv2.destroyAllWindows()
