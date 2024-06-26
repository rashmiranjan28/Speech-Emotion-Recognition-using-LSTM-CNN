
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
