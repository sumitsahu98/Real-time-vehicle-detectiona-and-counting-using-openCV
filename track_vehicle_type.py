import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO("yolov5s.pt")

# Video source
video_source = 'video.mp4'
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Cannot open video source")
    exit()

# Frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Parameters for rectangle size (for additional ROIs)
min_width_rectangle = 70
min_height_rectangle = 70
max_width_rectangle = 200
max_height_rectangle = 250

# Counting line positions
count_line_position = 530
offset = 6

# Left ROI parameters
roi_left_x_start_percent_upper = 0.3
roi_left_x_end_percent_upper = 0.515
roi_left_x_start_percent_lower = -0.5
roi_left_x_end_percent_lower = 0.5

# Right ROI parameters
roi_right_x_start_percent_upper = 0.515
roi_right_x_end_percent_upper = 0.7
roi_right_x_start_percent_lower = 0.5
roi_right_x_end_percent_lower = 1.5

# ROI Y positions
roi_y_start_percent = 0.5
roi_y_end_percent = 0.95

# Vehicle classes (COCO)
vehicle_classes = [2, 3, 6, 7]

# Counters
vehicle_count_left = 0
vehicle_count_right = 0
detected_vehicles = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error in reading frames.")
        break

    # Calculate ROI positions in pixels
    roi_left_x_start_upper = int(roi_left_x_start_percent_upper * frame_width)
    roi_left_x_end_upper = int(roi_left_x_end_percent_upper * frame_width)
    roi_left_x_start_lower = int(roi_left_x_start_percent_lower * frame_width)
    roi_left_x_end_lower = int(roi_left_x_end_percent_lower * frame_width)

    roi_right_x_start_upper = int(roi_right_x_start_percent_upper * frame_width)
    roi_right_x_end_upper = int(roi_right_x_end_percent_upper * frame_width)
    roi_right_x_start_lower = int(roi_right_x_start_percent_lower * frame_width)
    roi_right_x_end_lower = int(roi_right_x_end_percent_lower * frame_width)

    roi_y_start = int(roi_y_start_percent * frame_height)
    roi_y_end = int(roi_y_end_percent * frame_height)

    # Detect vehicles using YOLO
    start_time = time.time()
    results = model.predict(frame, stream=False)

    # Define ROIs
    points_left_roi = np.array([
        (roi_left_x_start_upper, roi_y_start),
        (roi_left_x_end_upper, roi_y_start),
        (roi_left_x_end_lower, roi_y_end),
        (roi_left_x_start_lower, roi_y_end)
    ], dtype=np.int32)

    points_right_roi = np.array([
        (roi_right_x_start_upper, roi_y_start),
        (roi_right_x_end_upper, roi_y_start),
        (roi_right_x_end_lower, roi_y_end),
        (roi_right_x_start_lower, roi_y_end)
    ], dtype=np.int32)

    # Draw ROIs
    # cv2.polylines(frame, [points_left_roi], isClosed=True, color=(255, 0, 0), thickness=2)
    # cv2.polylines(frame, [points_right_roi], isClosed=True, color=(0, 255, 255), thickness=2)

    # Draw counting lines
    cv2.line(frame, (25, count_line_position), (roi_left_x_end_upper, count_line_position), (255, 255, 255), 2)
    cv2.line(frame, (roi_right_x_start_upper, count_line_position), (frame_width - 25, count_line_position), (255, 255, 255), 2)

    # Parse YOLO results
    for result in results:
        boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])

        if class_id in vehicle_classes:
            # Calculate centroid of the bounding box
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            centroid = (centroid_x, centroid_y)

            # Check if the centroid is within the left ROI
            if cv2.pointPolygonTest(points_left_roi, centroid, False) >= 0:
                if count_line_position - offset <= centroid_y <= count_line_position + offset:
                    vehicle_id = f"{x1}_{y1}_{x2}_{y2}_left"
                    if vehicle_id not in detected_vehicles:
                        detected_vehicles.add(vehicle_id)
                        vehicle_count_left += 1

                # Draw bounding box and label for left ROI vehicles
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the centroid is within the right ROI
            elif cv2.pointPolygonTest(points_right_roi, centroid, False) >= 0:
                if count_line_position - offset <= centroid_y <= count_line_position + offset:
                    vehicle_id = f"{x1}_{y1}_{x2}_{y2}_right"
                    if vehicle_id not in detected_vehicles:
                        detected_vehicles.add(vehicle_id)
                        vehicle_count_right += 1

                # Draw bounding box and label for right ROI vehicles
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display counts
    cv2.putText(frame, f"Left Side Count: {vehicle_count_left}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Right Side Count: {vehicle_count_right}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Vehicle Detection and Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate FPS
    elapsed_time = time.time() - start_time
    print(f"FPS: {1 / elapsed_time:.2f}")

cap.release()
cv2.destroyAllWindows()
