import cv2
import numpy as np

# Video source (use 0 for webcam or 'video.mp4' for a video file)
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Parameters for rectangle size
min_width_rectangle = 70
min_height_rectangle = 70
max_width_rectangle = 250  # Set maximum width
max_height_rectangle = 250  # Set maximum height

# Line position and offset
count_line_position = 550
offset = 6

# Background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Function to calculate center point of a rectangle
def center_point(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)

detected_centers = []
tracked_objects = {}
vehicle_count = 0
object_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Adjust counting line position dynamically
    count_line_position = min(count_line_position, frame_height - 50)

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = algo.apply(blur)
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Contour detection
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw counting line
    cv2.line(frame, (25, count_line_position), (frame_width - 25, count_line_position), (0, 255, 0), 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Validate rectangle size for filtering
        if not (min_width_rectangle <= w <= max_width_rectangle and 
                min_height_rectangle <= h <= max_height_rectangle):
            continue

        # Calculate center point
        center = center_point(x, y, w, h)

        # Check if object is crossing the line
        if count_line_position - offset <= center[1] <= count_line_position + offset:
            # Avoid duplicate counting using object tracking
            if center not in tracked_objects.values():
                tracked_objects[object_id] = center
                object_id += 1
                vehicle_count += 1
                print(f"Vehicle Count: {vehicle_count}")

        # Draw rectangle and center point
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    # Update display with count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show processed frame
    cv2.imshow("Vehicle Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
