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

## 📂 Dataset: First Impressions V2

This project uses the **[First Impressions V2 Dataset](https://chalearnlap.cvc.uab.cat/dataset/24/description/)**, a publicly available multimodal dataset designed to support automatic personality recognition tasks from short video clips. The dataset provides annotations based on the **Big Five Personality Traits**—Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism.



## 📋 Contents of the Dataset

Each video sample in the dataset includes:

- 🎥 **Visual Data**: Short clips of individuals talking (used for facial feature,speech and body language extraction).
- 📝 **Transcripts**: Human-generated or machine-generated transcripts of speech (used for textual and linguistic analysis).
- 🧠 **Personality Labels**:Values representing the degree to which each of the Big Five traits is present
