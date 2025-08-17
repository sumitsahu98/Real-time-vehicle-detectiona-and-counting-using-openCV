import cv2
import numpy as np
import time

# Video source (use 0 for webcam or 'video.mp4' for a video file)
cap = cv2.VideoCapture('video.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Parameters for rectangle size
min_width_rectangle = 70
min_height_rectangle = 70
max_width_rectangle = 220
max_height_rectangle = 250

# Line position and offset
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

# ROI Y positions (common for both sides)
roi_y_start_percent = 0.5
roi_y_end_percent = 0.95

# Background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Function to calculate the center of a rectangle
def center_point(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)

# Initialize tracking and counters
tracked_objects_left = {}
tracked_objects_right = {}
vehicle_count_left = 0
vehicle_count_right = 0
object_id_left = 1
object_id_right = 1

# Flags for line color
line_color_left_is_green = False
line_color_right_is_green = False
line_color_left_reset_time = 0
line_color_right_reset_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

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

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = algo.apply(blur)
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Detect contours
    _,cleaned_mask=cv2.threshold(cleaned_mask,254,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define ROIs (trapezoidal)
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
    cv2.polylines(frame, [points_left_roi], isClosed=True, color=(255, 0, 0), thickness=2)  # Left ROI
    cv2.polylines(frame, [points_right_roi], isClosed=True, color=(0, 255, 255), thickness=2)  # Right ROI

    # Reset line colors after a short delay
    if line_color_left_is_green and time.time() > line_color_left_reset_time:
        line_color_left_is_green = False
    if line_color_right_is_green and time.time() > line_color_right_reset_time:
        line_color_right_is_green = False

    # Draw counting lines
    cv2.line(frame, (25, count_line_position), (roi_left_x_end_upper, count_line_position), (0, 255, 0) if line_color_left_is_green else (255, 255, 255), 2)
    cv2.line(frame, (roi_right_x_start_upper, count_line_position), (frame_width - 25, count_line_position), (0, 255, 0) if line_color_right_is_green else (255, 255, 255), 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on size
        if not (min_width_rectangle <= w <= max_width_rectangle and min_height_rectangle <= h <= max_height_rectangle):
            continue

        center = center_point(x, y, w, h)

        # Check left ROI
        if cv2.pointPolygonTest(points_left_roi, center, False) >= 0:
            if count_line_position - offset <= center[1] <= count_line_position + offset:
                if center not in tracked_objects_left.values():
                    tracked_objects_left[object_id_left] = center
                    object_id_left += 1
                    vehicle_count_left += 1
                    line_color_left_is_green = True
                    line_color_left_reset_time = time.time() + 0.2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # Check right ROI
        elif cv2.pointPolygonTest(points_right_roi, center, False) >= 0:
            if count_line_position - offset <= center[1] <= count_line_position + offset:
                if center not in tracked_objects_right.values():
                    tracked_objects_right[object_id_right] = center
                    object_id_right += 1
                    vehicle_count_right += 1
                    line_color_right_is_green = True
                    line_color_right_reset_time = time.time() + 0.2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    # Display vehicle counts
    cv2.putText(frame, f"Left Side Vehicles Count: {vehicle_count_left}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Right Side Vehicles Count: {vehicle_count_right}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Vehicle Counting", frame)

    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
