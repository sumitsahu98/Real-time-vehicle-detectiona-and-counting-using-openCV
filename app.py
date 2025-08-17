from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

# Global counters
vehicle_count_left = 0
vehicle_count_right = 0
count_lock = threading.Lock()

# Video source
cap = cv2.VideoCapture('video.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Parameters
min_width_rectangle = 70
min_height_rectangle = 70
max_width_rectangle = 220
max_height_rectangle = 250
count_line_position = 530
offset = 6

# Left ROI %
roi_left_x_start_percent_upper = 0.3
roi_left_x_end_percent_upper = 0.515
roi_left_x_start_percent_lower = -0.5
roi_left_x_end_percent_lower = 0.5

# Right ROI %
roi_right_x_start_percent_upper = 0.515
roi_right_x_end_percent_upper = 0.7
roi_right_x_start_percent_lower = 0.5
roi_right_x_end_percent_lower = 1.5

roi_y_start_percent = 0.5
roi_y_end_percent = 0.95

# Background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Tracking
tracked_objects_left = {}
tracked_objects_right = {}
object_id_left = 1
object_id_right = 1

line_color_left_is_green = False
line_color_right_is_green = False
line_color_left_reset_time = 0
line_color_right_reset_time = 0


def center_point(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)


def generate_frames():
    global vehicle_count_left, vehicle_count_right
    global tracked_objects_left, tracked_objects_right
    global object_id_left, object_id_right
    global line_color_left_is_green, line_color_right_is_green
    global line_color_left_reset_time, line_color_right_reset_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # ROI pixel positions
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = algo.apply(blur)
        dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        _, cleaned_mask = cv2.threshold(cleaned_mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ROIs
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

        # Reset line colors
        if line_color_left_is_green and time.time() > line_color_left_reset_time:
            line_color_left_is_green = False
        if line_color_right_is_green and time.time() > line_color_right_reset_time:
            line_color_right_is_green = False

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if not (min_width_rectangle <= w <= max_width_rectangle and min_height_rectangle <= h <= max_height_rectangle):
                continue

            center = center_point(x, y, w, h)

            # LEFT ROI
            if cv2.pointPolygonTest(points_left_roi, center, False) >= 0:
                if count_line_position - offset <= center[1] <= count_line_position + offset:
                    if center not in tracked_objects_left.values():
                        with count_lock:
                            tracked_objects_left[object_id_left] = center
                            vehicle_count_left += 1
                            object_id_left += 1
                            line_color_left_is_green = True
                            line_color_left_reset_time = time.time() + 0.2

            # RIGHT ROI
            elif cv2.pointPolygonTest(points_right_roi, center, False) >= 0:
                if count_line_position - offset <= center[1] <= count_line_position + offset:
                    if center not in tracked_objects_right.values():
                        with count_lock:
                            tracked_objects_right[object_id_right] = center
                            vehicle_count_right += 1
                            object_id_right += 1
                            line_color_right_is_green = True
                            line_color_right_reset_time = time.time() + 0.2

        # Overlay counts
        with count_lock:
            cv2.putText(frame, f"Left: {vehicle_count_left}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Right: {vehicle_count_right}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------- ROUTES ----------
@app.route('/')
def login():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/count')
def count():
    return render_template('count.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    with count_lock:
        return jsonify(left=vehicle_count_left, right=vehicle_count_right)


if __name__ == "__main__":
    app.run(debug=True)
