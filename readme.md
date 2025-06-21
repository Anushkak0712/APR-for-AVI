# Automatic Personality Recognition from Asynchronous Video Interviews

This project implements a multimodal deep learning pipeline for predicting Big Five personality traits from asynchronous video interviews. It processes audio, visual, and textual data to output trait scores, enabling scalable personality assessment for recruitment and research use cases.

## 🚀 Features

- 🎥 Processes uploaded video interviews (not live feed).
- 🔊 Extracts MFCC features from audio using Librosa.
- 🖼️ Samples and preprocesses visual frames using OpenCV and VGG16.
- 📝 Transcribes speech using OpenAI Whisper and encodes text with BERT.
- 🤖 Trains a multimodal deep learning model (TensorFlow) to predict Big Five traits.
- 📊 Outputs personality scores for Extraversion, Neuroticism, Agreeableness, Conscientiousness, and Openness.
- 🗃️ Supports batch inference and result logging.

## 🧠 Model Overview

- The model architecture is a late-fusion neural network combining:

- Visual Stream: VGG16 + LSTM on sampled frames

- Audio Stream: CNN on MFCC features

- Text Stream: Dense layers on BERT embeddings

- Fusion Layer: Concatenated output passed to Dense → Output (5 trait scores)


## 🛠 Tech Stack

### 🔗 Frontend
- React.js
- Material-UI (MUI)

### 🔧 Backend
- Flask
- Python
- PyMongo

### 🤖 Machine Learning & Data Processing
- TensorFlow (Keras)
- OpenCV
- Librosa
- BERT (Transformers)
- Whisper
- FFmpeg

### 🗄️ Database
- MongoDB Atlas
