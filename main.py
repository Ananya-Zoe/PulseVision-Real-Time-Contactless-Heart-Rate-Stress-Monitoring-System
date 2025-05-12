import streamlit as st
import subprocess

st.set_page_config(page_title="PulseVision")
st.title("Heart Rate and Stress Monitoring System")
st.write("Choose a function to run:")

col1, col2 = st.columns([1, 1]) 
with col1:
    st.subheader("Heart Rate Detection")  
    st.write("Detect your heart rate in real-time using facial analysis.")  
    if st.button("Start Heart Rate Detection"):
        st.write("Launching Heart Rate Detection...")
        subprocess.Popen(["streamlit", "run", "heart_rate/GUI.py"])

with col2:
    st.subheader("Stress Measurement")  
    st.write("Analyze stress levels based on facial expressions and eyebrow movement.")
    if st.button("Start Stress Measurement"):
        st.write("Launching Stress Measurement...")
        subprocess.Popen(["streamlit", "run", "stress/Code/eyebrow_detection.py"])