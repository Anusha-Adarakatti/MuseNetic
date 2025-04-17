Collecting workspace information# Music Generation using LSTM Neural Networks

This project implements a music generation system using Long Short-Term Memory (LSTM) neural networks. The model is trained on Chopin's compositions and can generate new piano melodies in a similar style.

## Project Overview

The system processes MIDI files of Chopin's music, extracts musical patterns (notes and chords), and trains a deep learning model to predict subsequent notes. The trained model can then generate new, original compositions with similar stylistic characteristics.

## Prerequisites

The project requires the following dependencies

```python
tensorflow
numpy
pandas
collections
IPython
music21
matplotlib
scikit-learn
seaborn
```

## Dataset

The project uses MIDI files of Chopin's compositions located in the `..inputclassical-music-midichopin` directory.

## How It Works

1. Data Processing
   - MIDI files are loaded and parsed to extract notes and chords
   - Rare notes (appearing less than 100 times) are filtered out
   - Notes are encoded into numerical representations

2. Model Architecture
   - Two-layer LSTM network with dropout layers
   - First LSTM layer 512 units with return sequences
   - Second LSTM layer 256 units
   - Dense layers for classification

3. Training
   - Sequences of 40 notes are used as input
   - The model predicts the next note in the sequence
   - Trained for 200 epochs with Adamax optimizer

4. Music Generation
   - Uses seed sequences from the validation set
   - Generates one note at a time, feeding each new note back into the model
   - Converts predicted note sequences back to MIDI format

## Usage

1. Load the notebook in a Jupyter environment
2. Run all cells sequentially
3. The model training takes place in the designated cell
4. Use the `Malody_Generator(Note_Count)` function to generate new musical pieces
5. Listen to the generated music through IPython's Audio display

## Sample Output

The generated music is available in both visual (sheet music) and audio format.
The model can generate compositions of any length by specifying the number of notes to produce.

## Files

- music-generation-lstm.ipynb Main Jupyter notebook containing all code
- `Melody_Generated.mid` MIDI file of generated melody
- Audio files of generated melodies in WAV format

---

Note This project is part of a natural language processing (NLP) study exploring generative models for sequential data.