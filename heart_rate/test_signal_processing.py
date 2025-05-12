# test_signal_processing.py
import cv2
import time
import face_utilities as fu

fu = fu.FaceUtilities()
cap = cv2.VideoCapture(0)

start_time = time.time()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        ret_process = fu.no_age_gender_face_process(frame, "68")
        if ret_process is not None:
            (frame, roi, mask) = ret_process
            if roi is not None and mask is not None:
                mean_val = fu.get_mean_rgb(roi, mask)
                fu.append_raw_data(mean_val)
                fu.draw_on_frame(frame, roi, mask)

                if fu.buffer_full():
                    print("Processing signal...")
                    signal = fu.get_signal_filtered()
                    heart_rate = fu.get_heart_rate()
                    if heart_rate:
                        print("Heart Rate: {:.2f} bpm".format(heart_rate))
                    fu.plot_signal_graph(signal)
    
        cv2.imshow("Heart Rate Monitor", frame)
    except Exception as e:
        print("Error in processing loop:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
cap.release()
cv2.destroyAllWindows()
print("Total execution time: {:.2f} seconds".format(end_time - start_time))
