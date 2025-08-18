import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from threading import Thread

# Vehicle counting class
class VehicleCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counting Application")

        # Create UI components
        self.label = tk.Label(root, text="Select a video file to start vehicle counting", font=("Arial", 14))
        self.label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start Counting", width=20, command=self.start_counting)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop", width=20, command=self.stop_counting, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.quit_button = tk.Button(root, text="Quit", width=20, command=root.quit)
        self.quit_button.pack(pady=20)

        # Vehicle counting parameters
        self.cap = None
        self.running = False

    def start_counting(self):
        # Open file dialog to select a video file
        video_file = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4")])

        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Start vehicle counting in a separate thread
            counting_thread = Thread(target=self.run_vehicle_counting)
            counting_thread.daemon = True
            counting_thread.start()

    def stop_counting(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def run_vehicle_counting(self):
        # Vehicle counting logic here
        vehicle_count_left = 0
        vehicle_count_right = 0
        object_id_left = 1
        object_id_right = 1

        # Parameters for rectangle size and ROIs
        min_width_rectangle = 70
        min_height_rectangle = 70
        max_width_rectangle = 220
        max_height_rectangle = 250

        count_line_position = 530
        offset = 6

        roi_left_x_start_percent_upper = 0.3
        roi_left_x_end_percent_upper = 0.515
        roi_left_x_start_percent_lower = -0.5
        roi_left_x_end_percent_lower = 0.5

        roi_right_x_start_percent_upper = 0.515
        roi_right_x_end_percent_upper = 0.7
        roi_right_x_start_percent_lower = 0.5
        roi_right_x_end_percent_lower = 1.5

        roi_y_start_percent = 0.5
        roi_y_end_percent = 0.95

        algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

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
            _, cleaned_mask = cv2.threshold(cleaned_mask, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw ROIs
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

            cv2.polylines(frame, [points_left_roi], isClosed=True, color=(255, 0, 0), thickness=2)  # Left ROI
            cv2.polylines(frame, [points_right_roi], isClosed=True, color=(0, 255, 255), thickness=2)  # Right ROI

            # Counting logic goes here (same as your existing code)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not (min_width_rectangle <= w <= max_width_rectangle and min_height_rectangle <= h <= max_height_rectangle):
                    continue

                center = (x + w // 2, y + h // 2)

                if cv2.pointPolygonTest(points_left_roi, center, False) >= 0:
                    if count_line_position - offset <= center[1] <= count_line_position + offset:
                        vehicle_count_left += 1

                if cv2.pointPolygonTest(points_right_roi, center, False) >= 0:
                    if count_line_position - offset <= center[1] <= count_line_position + offset:
                        vehicle_count_right += 1

            # Update vehicle count on the screen
            cv2.putText(frame, f"Left Side Vehicles Count: {vehicle_count_left}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Right Side Vehicles Count: {vehicle_count_right}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show the frame
            cv2.imshow("Vehicle Counting", frame)

            # Stop the program if 'q' is pressed
            if cv2.waitKey(4) & 0xFF == ord('q'):
                break

        # Release the capture after the loop ends
        self.cap.release()
        cv2.destroyAllWindows()

# Create the Tkinter root window and run the application
root = tk.Tk()
app = VehicleCountingApp(root)
root.mainloop()
