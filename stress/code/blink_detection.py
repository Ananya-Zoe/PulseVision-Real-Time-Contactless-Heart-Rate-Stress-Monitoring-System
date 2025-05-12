# blink_detection.py
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np

class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consec_frames=3):
        """
        Initializes the BlinkDetector with parameters:
        - ear_threshold: EAR below which an eye is considered closed.
        - consec_frames: Number of consecutive frames required to register a blink.
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.frame_counter = 0  # Counter for consecutive frames with eyes closed.
        self.blink_count = 0    # Total blink count.
        
        # Get the landmark indices for the left and right eyes from the 68-point model.
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    @staticmethod
    def compute_EAR(eye):
        """
        Computes the Eye Aspect Ratio (EAR) for a given eye.
        The EAR is defined as:
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Compute the euclidean distances between the vertical eye landmarks.
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Compute the euclidean distance between the horizontal eye landmarks.
        C = dist.euclidean(eye[0], eye[3])
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def update(self, facial_landmarks):
        """
        Update the blink detector with the facial landmarks from the current frame.
        - facial_landmarks: a NumPy array of shape (68, 2) containing facial landmark coordinates.
        
        Returns:
          (ear, blink_count): EAR computed for the current frame and updated blink count.
        """
        # Extract left and right eye coordinates from facial landmarks.
        leftEye = facial_landmarks[self.lStart:self.lEnd]
        rightEye = facial_landmarks[self.rStart:self.rEnd]
        leftEAR = BlinkDetector.compute_EAR(leftEye)
        rightEAR = BlinkDetector.compute_EAR(rightEye)
        # Average the EAR for both eyes.
        ear = (leftEAR + rightEAR) / 2.0

        # Check if the average EAR is below the threshold.
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            # If eyes have been closed for sufficient consecutive frames, register a blink.
            if self.frame_counter >= self.consec_frames:
                self.blink_count += 1
            # Reset the frame counter when the eyes are open.
            self.frame_counter = 0

        return ear, self.blink_count

# Standalone test if you run this file directly.
if __name__ == "__main__":
    print("Blink detection module loaded. You can import BlinkDetector from this module.")
