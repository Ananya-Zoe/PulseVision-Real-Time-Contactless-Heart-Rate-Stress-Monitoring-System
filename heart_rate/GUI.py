import cv2
import streamlit as st
import numpy as np
from webcam import Webcam
from process import Process
from video import Video 
import time
import matplotlib.pyplot as plt

# Initialize Webcam and Process Objects
webcam = Webcam()
process = Process()
video_processor = Video()

st.set_page_config(page_title="Heart Rate Detection", page_icon="❤")
# Streamlit UI
st.title("Heart Rate Monitor")

st.info("""Avoid sudden movements to prevent inaccurate readings. 
         Remain still for at least 30 seconds to improve measurement accuracy.
         Ensure you are seated in a well-lit environment for optimal performance.""")
col1, col2 = st.columns([2, 1]) 

with col1:
    st.subheader("Input: Webcam")
    video_placeholder = st.empty()
    frame_counter_placeholder = st.empty()
    frame_progress_bar = st.progress(0)
    bpm_placeholder = st.empty()
    frequency_placeholder = st.empty()
    

with col2:
    st.subheader("Signal & FFT Graphs")
    signal_plot_placeholder = st.empty()
    fft_plot_placeholder = st.empty()

    
# Initialize session state for storing results
if "last_bpm" not in st.session_state:
    st.session_state["last_bpm"] = None
    st.session_state["last_freq"] = None

input_type = st.radio("Select Input Source:", ("Webcam", "Upload Video"))

if input_type == "Webcam":
    # Start and Stop Monitoring Buttons
    if st.button("Start Monitoring"):
        webcam.start()
        process.reset()
        st.session_state["running"] = True

    if st.button("Stop Monitoring"):
        webcam.stop()
        st.session_state["running"] = False

    # Main Loop to Capture and Process Frames until buffer is full
    if "running" in st.session_state and st.session_state["running"]:
        while len(process.data_buffer) < process.buffer_size:
            frame = webcam.get_frame()
            if frame is None:
                st.warning("Webcam not accessible")
                break
            
            print("Frame captured!")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret_process = process.fu.no_age_gender_face_process(frame, "5")
            if ret_process:
                rects, _, _, aligned_face, aligned_shape = ret_process
                print("Face detected")
                for rect in rects:
                    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                left_cheek, right_cheek = process.extract_cheek_regions(aligned_face, aligned_shape)
                if left_cheek is not None and right_cheek is not None:
                    process.process_cheek_signals(left_cheek, right_cheek)
                    process.frame_in = aligned_face 
                    if process.run():
                        bpm = process.bpm
                        freq = process.freqs
                        print(f"Terminal Output: BPM = {bpm:.2f}")
                        st.session_state["last_bpm"] = f"{bpm:.2f} BPM"
                        st.session_state["last_freq"] = f"{freq[-1]:.2f} Hz" if len(freq) > 0 else "--"
                        bpm_placeholder.subheader(f"Heart Rate(BPM): {bpm:.2f}")
                        if len(process.freqs) > 0 and len(process.fft) > 0:
                            peak_idx = np.argmax(process.fft)
                            dominant_freq = process.freqs[peak_idx]
                            st.session_state["last_freq"] = f"{dominant_freq:.2f} Hz"
                            frequency_placeholder.subheader(f"Dominant Frequency: {dominant_freq:.2f} Hz")
                        else:
                            frequency_placeholder.subheader("Processing...")

                        fig1, ax1 = plt.subplots()
                        if len(process.samples) > 20:
                            ax1.plot(process.samples[-100:], color='green')
                            ax1.set_title("Signal Variation")
                            ax1.set_xlabel("Time")
                            ax1.set_ylabel("Amplitude")
                        else:
                            ax1.text(0.5, 0.5, "No Data Yet", ha='center', va='center', fontsize=12)
                        signal_plot_placeholder.pyplot(fig1)

                        fig2, ax2 = plt.subplots()
                        if len(process.freqs) > 0 and len(process.fft) > 0:
                            ax2.plot(process.freqs, process.fft, color='blue')
                            ax2.set_title("Processed Signal (FFT)")
                            ax2.set_xlabel("Frequency (Hz)")
                            ax2.set_ylabel("Power")
                        else:
                            ax2.text(0.5, 0.5, "No FFT Data Yet", ha='center', va='center', fontsize=12)
                        fft_plot_placeholder.pyplot(fig2)
                    else:
                        print("process.run() returned False! Still collecting data...")
            else:
                print("Face not detected in current frame.")
            
            st.image(frame_rgb, channels="RGB")
            video_placeholder.image(frame_rgb, channels="RGB")

            collected = len(process.data_buffer)
            total = process.buffer_size
            frame_counter_placeholder.markdown(f"*Frames Collected:* {collected}/{total}")
            frame_progress_bar.progress(min(collected / total, 1.0))
            time.sleep(0.01)

        print("Required number of samples collected. Stopping webcam.")
        webcam.stop()
        st.session_state["running"] = False

elif input_type == "Upload Video":
    st.info("Upload a video of length 10-15 seconds")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        temp_video_path = f"temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_video_path)

        if st.button("Start Video Processing"):
            video_processor.dirname = temp_video_path
            video_processor.start()
            process.reset()
            st.session_state["running"] = True

            while len(process.data_buffer) < process.buffer_size:
                frame = video_processor.get_frame()
                if frame is None:
                    st.warning("End of video processing.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                process.frame_in = frame_rgb

                if process.run():
                    bpm = process.bpm
                    freq = process.freqs
                    st.session_state["last_bpm"] = f"{bpm:.2f} BPM"
                    st.session_state["last_freq"] = f"{freq[-1]:.2f} Hz" if len(freq) > 0 else "--"
                    bpm_placeholder.subheader(f"Heart Rate(BPM): {bpm:.2f}")
                    if len(process.freqs) > 0 and len(process.fft) > 0:
                        peak_idx = np.argmax(process.fft)
                        dominant_freq = process.freqs[peak_idx]
                        st.session_state["last_freq"] = f"{dominant_freq:.2f} Hz"
                        frequency_placeholder.subheader(f"Dominant Frequency: {dominant_freq:.2f} Hz")
                    else:
                        frequency_placeholder.subheader("Processing...")
                    
                    fig1, ax1 = plt.subplots()
                    if len(process.samples) > 20:
                        ax1.plot(process.samples[-100:], color='green')
                        ax1.set_title("Signal Variation")
                        ax1.set_xlabel("Time")
                        ax1.set_ylabel("Amplitude")
                    else:
                        ax1.text(0.5, 0.5, "No Data Yet", ha='center', va='center', fontsize=12)
                    signal_plot_placeholder.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    if len(process.freqs) > 0 and len(process.fft) > 0:
                        ax2.plot(process.freqs, process.fft, color='blue')
                        ax2.set_title("Processed Signal (FFT)")
                        ax2.set_xlabel("Frequency (Hz)")
                        ax2.set_ylabel("Power")
                    else:
                        ax2.text(0.5, 0.5, "No FFT Data Yet", ha='center', va='center', fontsize=12)
                    fft_plot_placeholder.pyplot(fig2)

                st.image(frame_rgb, channels="RGB")
                video_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.01)

            video_processor.stop()
            st.session_state["running"] = False

if not st.session_state.get("running", False) and st.session_state["last_bpm"] is not None:
    st.markdown("---")
    st.header("Final Monitoring Results")
    st.subheader(f"Heart Rate: {st.session_state['last_bpm']}")
    st.subheader(f"Frequency: {st.session_state['last_freq']}")