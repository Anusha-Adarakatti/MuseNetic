# Music Generation with Deep Learning

This project implements multiple approaches to algorithmic music generation using deep learning techniques. It includes two main models: an **LSTM-based model** for classical piano music generation in the style of Chopin, and a **Variational Autoencoder (VAE)** for jazz music generation.

## Table of Contents

- Overview
- Project Components
- Requirements
- Datasets
- Model Architectures
  - LSTM Model
  - VAE Model
- Training Process
- Music Generation
- Visualization
- Usage
- Results
- Acknowledgments

---

## Overview

This project explores different neural network architectures for music generation:

1. **LSTM Neural Networks** for classical piano music generation, learning patterns from Chopin's compositions.
2. **Convolutional Variational Autoencoder (CVAE)** for jazz music generation, learning a compressed latent representation of musical data.

Both approaches demonstrate different strengths in capturing musical structures and generating coherent compositions.

---

## Project Components

- `music-generation-lstm.ipynb`: Implements the LSTM model for classical music generation
- `music-vae-jazz.ipynb`: Implements the VAE model for jazz music generation

---

## Requirements

To run this project, you need the following dependencies:

```bash
pip install tensorflow tensorflow-addons numpy pandas matplotlib seaborn scikit-learn librosa music21 imageio ipython
```

Key libraries:

- **TensorFlow 2.x**: Core deep learning framework
- **music21**: For MIDI processing (LSTM model)
- **librosa**: For audio processing (VAE model)
- **matplotlib/seaborn**: For visualization

---

## Datasets

The project uses two different datasets:

1. **LSTM Model**: Chopin's MIDI files located in `../input/classical-music-midi/chopin/`
2. **VAE Model**: GTZAN Dataset for Music Genre Classification, specifically the jazz genre

```python
# LSTM model dataset path
classical_path = "../input/classical-music-midi/chopin/"

# VAE model dataset path
jazz_path = "../input/gtzan-dataset-music-genre-classification/Data/genres_original"
```

---

## Model Architectures

### LSTM Model

The LSTM-based model for classical music generation consists of:

1. **Input Processing**: MIDI files are parsed and converted to note sequences
2. **Model Structure**:
   - First LSTM layer: 512 units with return sequences
   - Dropout layer (0.1)
   - Second LSTM layer: 256 units
   - Dense layer: 256 units
   - Dropout layer (0.1)
   - Output layer: Dense with softmax activation

### VAE Model

The Convolutional Variational Autoencoder for jazz music consists of:

1. **Encoder**: 1D convolutional layers with ResNet-inspired blocks
2. **Latent Space**: 2D latent space for visualization and sampling
3. **Decoder**: 1D transposed convolutional layers to reconstruct audio

Key components include custom ResNet1DBlocks for both encoding and decoding pathways.

---

## Training Process

### LSTM Training

- Sequences of 40 notes are used as input features
- The model predicts the next note in the sequence
- Trained for 200 epochs with Adamax optimizer
- Loss function: Categorical cross-entropy

### VAE Training

- Audio samples are processed into spectrograms
- The model is trained for 20 epochs using Adam optimizer
- Loss combines reconstruction loss and KL divergence

---

## Music Generation

### LSTM Generation

The LSTM model generates music by:

- Starting with a seed sequence
- Predicting one note at a time
- Feeding each new note back into the model
- Converting the resulting sequence to MIDI format

### VAE Generation

The VAE model generates music by:

- Sampling from the learned latent space
- Decoding the sampled vector into audio
- Converting the output to playable audio format

---

## Visualization

Both models include visualization capabilities:

- **LSTM Model**: Displays generated music as sheet music using music21
- **VAE Model**: Shows waveforms of generated audio and creates GIF animations

---

## Usage

### LSTM Model

1. Open `music-generation-lstm.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. Use `Malody_Generator(Note_Count)` to generate new music
4. Listen to the output using IPython's Audio display

### VAE Model

1. Open `music-vae-jazz.ipynb` in Jupyter Notebook
2. Run cells to preprocess data and train the model
3. Generate new jazz music samples
4. Visualize and listen to the generated music

---

## Results

Both models successfully generate music with characteristics of their respective training data:

- **LSTM Model**: Creates piano compositions with stylistic elements similar to Chopin
- **VAE Model**: Generates jazz-like audio samples with appropriate timbral and rhythmic features

Output files include:

- MIDI files of generated classical music
- WAV files of generated jazz music
- Visualizations of music structure

---

## Acknowledgments

- **Classical Music Dataset**: Used for training the LSTM model
- **GTZAN Dataset**: Used for training the VAE model
- **TensorFlow**: Framework for building and training both models
- **music21**: Library for MIDI processing and visualization
- **librosa**: Library for audio processing and visualization

---

Explore different approaches to music generation and enjoy the unique compositions created by each model!
