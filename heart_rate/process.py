import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from imutils import face_utils

class Process:
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100      # Target buffer size for processing
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()
        self.last_time = time.time()

    def extract_cheek_regions(self, aligned_face, aligned_shape):
        if aligned_face is None or aligned_shape is None:
            return None, None
        try:
            left_cheek = aligned_face[20:50, 10:40]  
            right_cheek = aligned_face[20:50, 60:90]
            return left_cheek, right_cheek
        except Exception:
            return None, None
            
    def process_cheek_signals(self, left_cheek, right_cheek):
        if left_cheek is None or right_cheek is None:
            return None

        g_left = self.sp.extract_color(left_cheek)
        g_right = self.sp.extract_color(right_cheek)
        g = (g_left + g_right) / 2

        # Timestamp using elapsed time since t0
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)
        self.samples = np.array(self.data_buffer) 

        # Keep only the most recent samples up to the buffer_size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]

        return g

    def run(self):
        frame = self.frame_in
        ret_process = self.fu.no_age_gender_face_process(frame, "5")
        if ret_process is None:
            print("Face not detected in run(), skipping frame.")
            return True  # Continue collecting data

        _, _, _, aligned_face, aligned_shape = ret_process
        left_cheek, right_cheek = self.extract_cheek_regions(aligned_face, aligned_shape)
        if left_cheek is None or right_cheek is None:
            print("Cheek regions not detected, skipping frame.")
            return True  # Continue collecting data

        self.process_cheek_signals(left_cheek, right_cheek)

        # Only process once we have enough samples
        if len(self.data_buffer) < self.buffer_size:
            print(f"Collecting data: {len(self.data_buffer)}/{self.buffer_size} (Not enough data yet)")
            return True

        current_buffer = self.data_buffer[-self.buffer_size:]
        current_times = self.times[-self.buffer_size:]

        if len(current_times) > 1 and (current_times[-1] - current_times[0]) > 0:
            self.fps = float(len(current_buffer)) / (current_times[-1] - current_times[0])
        else:
            self.fps = 30

        print(f"FPS estimated: {self.fps:.2f}")

        # Instead of skipping processing if FPS is too low, use a fallback value.
        if self.fps < 5:
            print("FPS too low, using fallback FPS value for processing.")
            self.fps = 5  # Fallback FPS value

        # Interpolate data to ensure even spacing
        even_times = np.linspace(current_times[0], current_times[-1], len(current_buffer))
        processed = np.interp(even_times, current_times, np.array(current_buffer))

        windowed = np.hamming(len(processed)) * processed
        norm = windowed / np.linalg.norm(windowed)

        filtered = self.butter_bandpass_filter(norm, lowcut=0.8, highcut=3.0, fs=self.fps, order=4)

        raw_fft = np.fft.rfft(filtered)
        self.freqs = float(self.fps) / len(filtered) * np.arange(len(filtered) // 2 + 1)
        freqs_bpm = 60.0 * self.freqs
        self.fft = np.abs(raw_fft) ** 2

        idx = np.where((freqs_bpm > 50) & (freqs_bpm < 180))
        pruned = self.fft[idx]
        pfreq = freqs_bpm[idx]

        if len(pruned) > 0:
            idx_max = np.argmax(pruned)
            self.bpm = pfreq[idx_max]
            print(f"BPM Detected: {self.bpm:.2f} BPM")
        else:
            self.bpm = 0
            print("No BPM detected in this frame.")

        return True

    def reset(self):
        self.__init__()
        
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        if fs <= 0:
            return None, None
        
        nyq = 0.5 * fs
        low = max(0.01, min(lowcut / nyq, 0.99))
        high = max(low + 0.01, min(highcut / nyq, 0.99))
        
        if high <= low:
            return None, None
    
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return signal.lfilter(b, a, data) if b is not None and a is not None else data
