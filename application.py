import streamlit as st
import soundfile as sf
import numpy as np
from feat import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Label encoder
labelencoder = LabelEncoder()

# Load the saved model
model_path = 'LSTM_CNN_MODEL_2.h5'
model = load_model(model_path)

# Label mapping
label_mapping = {0: 'angry', 
                 1: 'apologetic', 
                 2: 'calm', 
                 3: 'excited', 
                 4: 'fear', 
                 5: 'happy', 
                 6: 'neutral', 
                 7: 'sad'}

# Set the title of the Streamlit app
st.title("Speech Emotion Recognition")

# File uploader for audio files
audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav"])

# Set the interval for segments
interval = st.number_input("Set the interval (1-15 seconds) for emotion detection segments:", min_value=1.0, max_value=15.0, value=3.0, step=0.1)

# Button to upload
if st.button("Upload"):
    if audio_file:
        # Read the audio file
        audio_data, samplerate = sf.read(audio_file)
        # Convert the audio file to WAV format and save it
        output_file_path = 'uploaded_audio.wav'
        sf.write(output_file_path, audio_data, samplerate)
        
        st.audio(audio_file)
    else:
        st.error("Please upload an audio file.")

# Function to process audio and predict emotions
def predict_emotions(audio_path, interval):
    audio_data, samplerate = sf.read(audio_path)
    duration = len(audio_data) / samplerate
    emotions = []

    for start in np.arange(0, duration, interval):
        end = start + interval
        if end > duration:
            end = duration
        segment = audio_data[int(start*samplerate):int(end*samplerate)]
        segment_path = 'segment.wav'
        sf.write(segment_path, segment, samplerate)
        feat = features_extractor(segment_path)
        feat = feat.reshape(1, -1)
        predictions = model.predict(feat)
        predicted_label = np.argmax(predictions, axis=1)
        emotions.append((start, end, label_mapping[predicted_label[0]]))
    
    return emotions

# Button to predict
if st.button("Predict"):
    if audio_file:
        emotions = predict_emotions('uploaded_audio.wav', interval=interval)
        
        # Create a DataFrame to display emotions
        emotions_df = pd.DataFrame(emotions, columns=["Start", "End", "Emotion"])
        st.write(emotions_df)
        
        # Save emotions to a log file
        log_file_path = 'emotion_log.csv'
        emotions_df.to_csv(log_file_path, index=False)
        
        # Extrapolate major emotions
        major_emotion = emotions_df['Emotion'].mode().values[0]
        st.write(f"Major emotion: {major_emotion}")
        
        st.success(f"Emotion log saved to {log_file_path}")
        
        # Add download button for the emotion log file
        with open(log_file_path, "rb") as file:
            btn = st.download_button(
                label="Download Emotion Log",
                data=file,
                file_name='emotion_log.csv',
                mime='text/csv'
            )
    else:
        st.error("Please upload an audio file.")

# Additional message at the bottom of the page
st.write("Thank you for using the app!")
