# Music VAE for music Generation

## Overview

This project implements a **Convolutional Variational Autoencoder (CVAE)** for generating Music music using deep learning techniques. The system learns to encode musical audio into a compact latent representation and then decode it back to generate new, unique Music pieces while preserving stylistic elements of the original data.

## Table of Contents

- [Music VAE for Music Generation](#music-vae-for-Music-generation)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Technical Background](#technical-background)
    - [Variational Autoencoders](#variational-autoencoders)
    - [Audio Processing](#audio-processing)
  - [Dataset](#dataset)
  - [Implementation Details](#implementation-details)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Architecture](#model-architecture)
      - [Resnet1DBlock](#resnet1dblock)
      - [CVAE Model](#cvae-model)
    - [Training Process](#training-process)
    - [Loss Function](#loss-function)
  - [Results](#results)
    - [Visualization](#visualization)
    - [Generated Music](#generated-music)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Future Work](#future-work)
  - [References](#references)

## Technical Background

### Variational Autoencoders

Variational Autoencoders (VAEs) are generative models that learn to encode input data into a lower-dimensional latent space and then reconstruct the original data from this compressed representation. Unlike traditional autoencoders, VAEs impose a regularization on the latent space by enforcing a prior distribution (typically Gaussian), which makes them suitable for generation tasks.

The VAE consists of two main components:

- **Encoder**: Compresses input data into a latent representation, outputting parameters (mean and variance) of the latent distribution
- **Decoder**: Reconstructs the original data from samples drawn from the latent distribution

For music generation, VAEs can capture high-level musical features in the latent space, allowing us to generate new music by sampling from this space.

### Audio Processing

This project uses raw waveform audio data sampled at 3000 Hz for training the model. Each audio sample is represented as a 1D array of length 90,001, corresponding to 30 seconds of audio. The waveform representation preserves all temporal and frequency information of the original audio signals.

## Dataset

The project uses the **GTZAN Dataset for Music Genre Classification**, specifically focusing on the Music genre. The dataset contains 100 30-second audio clips for each of 10 music genres. The Music subset is used exclusively for this project.

Dataset path configuration:

```python
DATASET = 'gtzan-dataset-music-genre-classification'
```

For the purpose of this experiment, the Music tracks are split into training and testing sets as follows:

- Training set: 13 tracks
- Testing set: 12 tracks

## Implementation Details

### Data Preprocessing

The data preprocessing pipeline includes:

1. **Loading audio files**: Using `librosa` to load audio files with a fixed sampling rate (3000 Hz) and duration (30 seconds)
2. **Reshaping**: Converting each audio sample to a shape of (1, 90001) for model input
3. **Creating TensorFlow datasets**: Converting the data into TensorFlow dataset objects with appropriate batching and shuffling

Key data preparation functions:

- `DatasetLoader`: Splits the Music tracks into training and testing sets
- `load`: Loads audio files with specified parameters
- `map_data`: Maps the loading function to TensorFlow dataset

### Model Architecture

The model architecture is based on a convolutional neural network with ResNet-inspired blocks to capture both local and global audio patterns.

#### Resnet1DBlock

A custom 1D ResNet block is implemented with:

- For encoding: 1D convolutions with instance normalization
- For decoding: 1D transposed convolutions with batch normalization
- Skip connections to preserve information flow
- LeakyReLU activations

```python
class Resnet1DBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, type='encode'):
        super(Resnet1DBlock, self).__init__(name='')

        if type=='encode':
            self.conv1a = layers.Conv1D(filters, kernel_size, 2, padding="same")
            self.conv1b = layers.Conv1D(filters, kernel_size, 1, padding="same")
            self.norm1a = tfa.layers.InstanceNormalization()
            self.norm1b = tfa.layers.InstanceNormalization()
        if type=='decode':
            self.conv1a = layers.Conv1DTranspose(filters, kernel_size, 1, padding="same")
            self.conv1b = layers.Conv1DTranspose(filters, kernel_size, 1, padding="same")
            self.norm1a = tf.keras.layers.BatchNormalization()
            self.norm1b = tf.keras.layers.BatchNormalization()
```

#### CVAE Model

The CVAE model consists of:

**Encoder:**

- Input layer for (1, 90001) shaped data
- Multiple Conv1D layers with increasing filter sizes (64 → 128 → 128 → 256)
- ResNet blocks after each convolution
- Flatten layer
- Dense output layer producing latent parameters (mean and logvar)

**Decoder:**

- Input layer for latent vectors
- Reshape to (1, latent_dim)
- Multiple transposed Conv1D layers with decreasing filter sizes (512 → 256 → 128 → 64)
- ResNet blocks between convolutions
- Final Conv1DTranspose layer to reconstruct the original audio

**Key Methods:**

- `encode`: Encodes input to latent space parameters
- `reparameterize`: Implements the reparameterization trick for sampling
- `decode`: Reconstructs audio from latent vectors
- `sample`: Generates new samples from random points in the latent space

```python
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([...])
        self.decoder = tf.keras.Sequential([...])
```

### Training Process

The training process is implemented with these components:

1. **Optimizer**: Adam optimizer with learning rate 0.0003
2. **Loss calculation**:
   - Reconstruction loss using binary cross-entropy
   - KL divergence loss to regularize the latent space
3. **Training loop**:
   - Runs for 20 epochs
   - Uses gradient tape for automatic differentiation
   - Updates model parameters using backpropagation
   - Tracks and displays ELBO (Evidence Lower BOund) metric
   - Generates and saves sample images after each epoch

```python
def train(train_dataset, test_dataset, model, save):
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_x = np.asarray(train_x)[0]
            train_step(model, train_x, optimizer)
        # Evaluation and visualization code...
```

### Loss Function

The VAE loss function has two components:

1. **Reconstruction Loss**: Measures how well the decoder reconstructs the input audio

   ```python
   reconstruction_loss = tf.reduce_mean(
       tf.keras.losses.binary_crossentropy(x, x_logit)
   )
   ```

2. **KL Divergence Loss**: Regularizes the latent space to follow a standard normal distribution
   ```python
   loss_KL = -tf.reduce_mean(logpx_z + logpz - logqz_x)
   ```

The total loss is the sum of these two components:

```python
total_loss = reconstruction_loss + loss_KL
```

## Results

### Visualization

The training process visualizes:

1. **Waveform plots**: The generated audio is visualized as waveforms using `librosa.display.waveplot`
2. **Training progression**: A GIF animation is created showing how generated samples improve over epochs

Visualization code:

```python
def generate_and_save_images(model, epoch, test_sample, save):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(18, 15))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        wave = np.asarray(predictions[i])
        librosa.display.waveplot(wave[0], sr=3000)

    plt.savefig('{}_{:04d}.png'.format(save, epoch))
    plt.show()
```

### Generated Music

After training, the model can generate new Music music samples by:

1. Encoding existing music to the latent space
2. Perturbing the latent representation
3. Decoding back to audio space

The notebook provides functionality to play the generated audio samples directly:

```python
def inference(test_dataset, model):
    save_music = []
    for test in test_dataset:
        mean, logvar = model.encode(test)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        for pred in predictions:
            wave = np.asarray(pred)
            save_music.append(wave)
    return save_music
```

## Dependencies

The project requires the following Python libraries:

```
tensorflow>=2.0.0
tensorflow-addons
numpy
pandas
matplotlib
librosa
imageio
ipython
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Anusha-Adarakatti/MuseNetic.git
   cd vae
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the GTZAN dataset and place it in the appropriate directory:
   ```
   /path/to/project/input/gtzan-dataset-music-genre-classification/Data/genres_original/Music/
   ```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook notebooks/music-vae-Music.ipynb
   ```

2. Run the cells in order to:

   - Load and preprocess the dataset
   - Define and initialize the model
   - Train the model
   - Generate new music samples

3. To generate new Music music after training:

   ```python
   # Load a saved model or use the trained model
   saved_musics = inference(test_dataset, model)

   # Play a generated music sample
   ipd.Audio(saved_musics[0][0], rate=3000)
   ```

## Future Work

Possible future improvements include:

1. **Larger dataset**: Train on more Music tracks for greater variety
2. **Higher latent dimensions**: Experiment with larger latent spaces for more expressive generation
3. **Conditional generation**: Extend the model to support conditioning on musical attributes
4. **Hierarchical models**: Implement hierarchical VAEs to capture structure at different time scales
5. **Evaluation metrics**: Develop quantitative metrics to evaluate the quality of generated music
6. **Genre transfer**: Extend the model to translate between different music genres
7. **User interface**: Create a user-friendly interface for music generation
