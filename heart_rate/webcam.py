import cv2
import numpy as np
import time

class Webcam(object):
    def __init__(self):
        self.dirname = ""  # For uniform input handling
        self.cap = None
    
    def start(self):
        print("[INFO] Start webcam")
        time.sleep(1)  # Wait for the camera to be ready
        self.cap = cv2.VideoCapture(0)
        self.valid = False
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
    
    def get_frame(self):
        if self.valid:
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Stop webcam")
    
    def get_fps(self):
        """Returns the FPS of the webcam. If unavailable, estimates it dynamically."""
        if self.cap is None or not self.cap.isOpened():
            return 30  # Default FPS if the camera is not opened

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:  # If OpenCV fails to get FPS, estimate it
            return self.estimate_fps()
        return fps

    def estimate_fps(self):
        """Estimates FPS dynamically by capturing frames for 1 second."""
        frame_count = 0
        start_time = time.time()

        for _ in range(10):  # Capture 10 frames to estimate FPS
            ret, _ = self.cap.read()
            if not ret:
                break
            frame_count += 1

        elapsed_time = time.time() - start_time
        return frame_count / elapsed_time if elapsed_time > 0 else 30  # Default to 30 FPS if error
