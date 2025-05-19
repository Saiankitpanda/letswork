import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment, effects
import noisereduce as nr
import librosa.display

# Load the saved model
model_path = 'saved_models/Emotion_Voice_Detection_Model.h5'
model = load_model(model_path)

# Load the label encoder
lb = LabelEncoder()
lb.classes_ = np.load('label_encoder_classes.npy')
print("Label encoder classes:", lb.classes_)  # Debugging: Check the classes

# Function to extract features from audio
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate), axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate), axis=0)
        features = np.hstack([mfccs, chroma, mel, contrast])  # Combine features

        # Ensure the number of features matches the model's expected input shape
        if len(features) > 216:  # Truncate if necessary
            features = features[:216]
        elif len(features) < 216:  # Pad if necessary
            features = np.pad(features, (0, 216 - len(features)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None
    return features

# Function to predict emotion from audio
def predict_emotion(filename):
    features = extract_features(filename)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=2)  # Add channel dimension

        preds = model.predict(features, batch_size=32, verbose=1)
        print("Raw predictions:", preds)  # Debugging

        # Print sorted probabilities for better debugging
        sorted_indices = np.argsort(preds[0])[::-1]
        confidence_scores = {lb.classes_[i]: float(preds[0][i]) for i in sorted_indices}

        preds1 = preds.argmax(axis=1)
        abc = preds1.astype(int).flatten()
        predictions = lb.inverse_transform(abc)
        return predictions[0], confidence_scores
    return None, None

# Function to plot waveforms
def plot_waveforms(file_path):
    # Load the audio file
    rawsound = AudioSegment.from_file(file_path)
    x, sr = librosa.load(file_path, sr=None)

    # Create tabs for each waveform
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Initial Audio", "Normalized Audio", "Trimmed Audio", "Padded Audio", "Noise-Reduced Audio"])

    with tab1:
        st.write("### Initial Audio Waveform")
        plt.figure(figsize=(12, 2))
        librosa.display.waveshow(x, sr=sr)
        st.pyplot(plt)

    with tab2:
        st.write("### Normalized Audio Waveform")
        normalizedsound = effects.normalize(rawsound, headroom=5.0)
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
        normal_x = normal_x / (2**15)  # Scale to [-1, 1]
        plt.figure(figsize=(12, 2))
        librosa.display.waveshow(normal_x, sr=sr)
        st.pyplot(plt)

    with tab3:
        st.write("### Trimmed Audio Waveform")
        xt, index = librosa.effects.trim(normal_x, top_db=45)
        plt.figure(figsize=(12, 2))
        librosa.display.waveshow(xt, sr=sr)
        st.pyplot(plt)

    with tab4:
        st.write("### Padded Audio Waveform")
        target_length = 173056
        if len(xt) < target_length:
            padded_x = np.pad(xt, (0, target_length - len(xt)), 'constant')
        else:
            padded_x = xt[:target_length]  # Truncate if longer than target_length
        plt.figure(figsize=(12, 2))
        librosa.display.waveshow(padded_x, sr=sr)
        st.pyplot(plt)

    with tab5:
        st.write("### Noise-Reduced Audio Waveform")
        final_x = nr.reduce_noise(y=padded_x, y_noise=padded_x, sr=sr)
        plt.figure(figsize=(12, 2))
        librosa.display.waveshow(final_x, sr=sr)
        st.pyplot(plt)

# Streamlit app
def main():
    st.title("Emotion Detection from Audio")
    st.write("Upload an audio file to predict the emotion.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display a loading spinner while predicting
        with st.spinner("Predicting emotion... Please wait..."):
            # Simulate a delay for demonstration purposes
            time.sleep(2)  # You can remove this line in production
            # Predict emotion
            emotion, confidence_scores = predict_emotion("temp_audio.wav")

        if emotion:
            st.success(f"Predicted Emotion: {emotion}")
            st.write("### Confidence Scores:")
            
            # Filter out confidence scores that are 0.0000
            filtered_confidence_scores = {k: v for k, v in confidence_scores.items() if v != 0.0000}
            
            # Display filtered confidence scores
            for emotion_label, score in filtered_confidence_scores.items():
                st.write(f"{emotion_label}: {score:.4f}")
            
            # Plot waveforms in separate tabs
            plot_waveforms("temp_audio.wav")
        else:
            st.error("Could not predict emotion.")

if __name__ == "__main__":
    main()