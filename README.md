Download this shape_predictor_68_face_landmarks.dat into heart_rate\stress and heart_rate\heart_rate
pip install dependencies.txt in the stress folder and pip install requirements.txt in the heart_rate folder. If they don't install, install them individually .
Run the following command, considering your main folder name is heart_rate
heart_rate\heart_rate> cd heart_rate
heart_rate\heart_rate\heart_rate> .\myenv\Scripts\activate 
heart_rate\heart_rate\heart_rate> cd .. 
heart_rate\heart_rate> streamlit run main.py
