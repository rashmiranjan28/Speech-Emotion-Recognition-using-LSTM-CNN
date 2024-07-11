import streamlit as st
import soundfile as sf
import numpy as np
from feat import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
import speech_recognition as sr
import wave


# Label encoder
labelencoder = LabelEncoder()

# Load the saved model
model_path = 'cnn_lstm.keras'
model = load_model(model_path)

# Label mapping
label_mapping = {0: 'angry',
                 1: 'excited',
                 2: 'fear',
                 3: 'happy',
                 4: 'neutral',
                 5: 'sad'}

# Set the title of the Streamlit app
st.title("Speech Emotion Recognition")

# File uploader for audio files
audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav"])

# Set the interval for segments
interval = st.number_input("Set the interval (0.00-15.00 seconds) for emotion detection segments:",
                           min_value=0.00, max_value=15.00, value=3.00, step=0.01)

# Button to upload
if st.button("Upload"):

    if audio_file:
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
        print()
        emotions = predict_emotions('uploaded_audio.wav', interval=interval)

        # Create a DataFrame to display emotions
        emotions_df = pd.DataFrame(
            emotions, columns=["Start", "End", "Emotion"])
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

        x = word_count1('uploaded_audio.wav')
        y = get_speaking_rate('uploaded_audio.wav')

        st.write(f'Number of words = {x[0]}')
        st.write(f'Transcript = {x[1]}')

        st.write(f'Speaking rate = {y} syllables per second')

    else:
        st.error("Please upload an audio file.")


# Additional message at the bottom of the page
st.write("Thank you for using the app!")

file_path = 'path/to/your/audio/file'
try:
    audio, sr = librosa.load(audio_file, sr=None)
except Exception as e:
    print(f"An error occurred: {e}")
