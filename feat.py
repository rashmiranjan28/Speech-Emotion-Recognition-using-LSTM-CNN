
import librosa
import numpy as np

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    
    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=25)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    # Extract Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_scaled_features = np.mean(zcr.T, axis=0)
    
    # Extract Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled_features = np.mean(chroma.T, axis=0)
    
    # Extract Mel Spectrogram Features
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_scaled_features = np.mean(mel.T, axis=0)
    
    # Concatenate all features into a single array
    features = np.hstack((mfccs_scaled_features, zcr_scaled_features, chroma_scaled_features, mel_scaled_features))
    
    return features


#########################################################################################################################
import speech_recognition as sr

def recognize_speech_from_file(audio_file_path):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file_path) as source:
        
        audio_data = recognizer.record(source)  # Read the entire audio file
        
        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio_data)
          
            return text
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.UnknownValueError:
            print("Could not understand the audio")

def count_words(text):
    words = text.split()
    return len(words)

def word_count(audio_path):
    transcript = recognize_speech_from_file(audio_file_path=audio_path)
    if transcript:
        return [count_words(transcript),transcript]

########################################################################################################################
import speech_recognition as sr
import wave

def recognize_speech_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_file_path)
    with audio_file as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def count_words(text):
    words = text.split()
    return len(words)

def get_audio_duration(audio_file_path):
    with wave.open(audio_file_path, 'r') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
    return duration

def word_count1(audio_path):
    transcript = recognize_speech_from_file(audio_file_path=audio_path)
    if transcript:
        duration = get_audio_duration(audio_path)
        return [count_words(transcript), transcript, duration]
    else:
        return [0, None, 0.0]
    
word_count('angry_Akash.wav')

# print(word_count1(r'c:\Users\hp\OneDrive\Desktop\Major Emotions\Mixed\Angry-1-3-1.wav'))
# Example usage
# audio_path = 'angry_Ansh.wav'
# result = word_count(audio_path)
# print(result)

import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS

def get_speaking_rate(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract speech segments
    segments = aS.silence_removal(y, sr, 0.020, 0.020, smooth_window=1.0, weight=0.3, plot=False)
    
    # Total speech duration
    speech_duration = sum([end - start for start, end in segments])
    
    # Number of syllables (approximation)
    num_syllables = len(librosa.effects.split(y, top_db=30))
    
    # Calculate speaking rate (syllables per second)
    speaking_rate = num_syllables / speech_duration if speech_duration > 0 else 0
    
    return speaking_rate

# Example usage
# file_path = 'angry_Ansh.wav'
# speaking_rate = get_speaking_rate(file_path)[0]
# print(f"Speaking Rate: {speaking_rate:.2f} syllables per second")
# print(get_speaking_rate(file_path)[1])
# print(get_speaking_rate(file_path)[2])

