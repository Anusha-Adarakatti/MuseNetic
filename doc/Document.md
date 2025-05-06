# MuseNetic: AI Music Generation System
## Project Documentation

## 1. Requirement Analysis Documentation

### Software Requirement Specification (SRS)

#### Project Overview
MuseNetic is an AI-powered music generation system that creates original musical compositions through two distinct approaches:
- Bidirectional LSTM networks for note-sequence generation (classical music)
- Variational Autoencoders (VAEs) for waveform-based generation (jazz and classical music)

#### Functional Requirements
1. **Data Processing**
   - Process MIDI files for classical music training
   - Process audio files for jazz and classical training
   - Filter rare notes and create data encodings
   - Create sequence data for model training

2. **Model Training**
   - Train bidirectional LSTM models on note sequences
   - Train VAE models on audio waveforms
   - Visualize model training progress
   - Save trained models for future use

3. **Music Generation**
   - Generate classical music using bidirectional LSTM
   - Generate jazz music using VAE
   - Generate classical music using VAE
   - Export generated music in appropriate formats (MIDI or WAV)

4. **Visualization**
   - Display sheet music notation of generated music
   - Display audio waveforms of generated music
   - Create animations showing model training progress

#### Non-Functional Requirements
1. **Performance**
   - Generate music within reasonable time constraints
   - Support training on consumer-grade hardware

2. **Usability**
   - Provide intuitive Jupyter notebook interfaces
   - Include clear documentation on usage

3. **Maintainability**
   - Organize code in modular components
   - Include comprehensive code comments

### User Stories

1. **As a music composer:**
   - I want to generate novel musical ideas in classical style
   - I want to visualize the generated music as sheet notation
   - I want to export the generated music as MIDI files

2. **As a music producer:**
   - I want to generate jazz-style audio segments
   - I want to hear various generated samples before selection
   - I want to export high-quality audio files of generated music

3. **As a researcher:**
   - I want to compare different music generation approaches
   - I want to visualize the training process
   - I want to experiment with different model architectures

4. **As a student:**
   - I want to understand how AI can learn musical patterns
   - I want to see the relationship between model parameters and output quality
   - I want examples of AI-generated music in different styles

### Stakeholder Analysis

#### Primary Stakeholders

1. **Musicians and Composers**
   - **Interests**: New creative tools, inspiration for compositions
   - **Needs**: High-quality, musically coherent output
   - **Concerns**: Originality of AI-generated content, ease of use

2. **Researchers and Students**
   - **Interests**: Understanding AI music generation techniques
   - **Needs**: Well-documented code, explainable results
   - **Concerns**: Educational value, theoretical soundness

3. **Music Producers**
   - **Interests**: Novel sounds and musical ideas
   - **Needs**: Diverse outputs, easy integration with DAWs
   - **Concerns**: Audio quality, uniqueness of generated material

#### Secondary Stakeholders

1. **Music Education Institutions**
   - **Interests**: Teaching tools for music composition
   - **Needs**: Educational resources on AI music
   - **Concerns**: Pedagogical value, student engagement

2. **Music Software Developers**
   - **Interests**: Potential API integration
   - **Needs**: Well-documented interfaces
   - **Concerns**: Compatibility, performance

## 2. System Design Documentation

### High-Level Design (HLD)

#### System Architecture
MuseNetic consists of three main modules:

1. **Bidirectional LSTM Module for Classical Music**
   - Processes MIDI files
   - Extracts and encodes notes
   - Trains sequence models
   - Generates note sequences
   - Converts outputs to MIDI format

2. **VAE Module for Jazz Music**
   - Processes audio waveforms
   - Trains VAE with ResNet blocks
   - Generates audio in jazz style
   - Provides audio playback and visualization

3. **VAE Module for Classical Music**
   - Similar architecture to Jazz VAE
   - Trained on classical music samples
   - Generates classical-style audio outputs

#### Component Diagram
```
┌───────────────────────────────────┐
│           Data Processing         │
├───────────┬─────────────┬─────────┤
│ MIDI      │ Jazz Audio  │Classical│
│ Processing│ Processing  │ Audio   │
└─────┬─────┴──────┬──────┴────┬────┘
      │            │           │
┌─────▼─────┐┌─────▼─────┐┌────▼─────┐
│Bidirectional│VAE Jazz  │ VAE       │
│ LSTM Model │ Model    │ Classical │
└─────┬─────┘└─────┬─────┘└────┬─────┘
      │            │           │
┌─────▼─────┐┌─────▼─────┐┌────▼─────┐
│ Classical │ Jazz Audio │ Classical │
│ MIDI Gen  │ Gen       │ Audio Gen │
└─────┬─────┘└─────┬─────┘└────┬─────┘
      │            │           │
┌─────▼────────────▼───────────▼─────┐
│       Visualization & Output       │
└───────────────────────────────────┘
```

### Low-Level Design (LLD)

#### 1. Bidirectional LSTM Module

**Data Processing Component**
- `extract_notes()`: Extracts notes and chords from MIDI files
- `chords_n_notes()`: Converts note representations to MIDI objects
- Feature encoding: Maps notes to numerical indices
- Sequence creation: Creates fixed-length sequences for training

**Model Architecture**
- Input Layer: Shape = (sequence_length, 1)
- Bidirectional LSTM Layer 1: 512 units with sequence return
- Dropout Layer 1: 0.3 dropout rate
- Bidirectional LSTM Layer 2: 256 units with sequence return
- Dropout Layer 2: 0.3 dropout rate
- Bidirectional LSTM Layer 3: 128 units
- Dense Layer 1: 256 units with ReLU activation
- Dropout Layer 3: 0.3 dropout rate
- Output Layer: Dense with softmax activation (size = vocabulary size)

**Generator Component**
- `Melody_Generator()`: Generates notes based on seed sequences
- Note sequence generation with temperature-based diversity
- Conversion to MIDI format for output

#### 2. VAE Modules

**ResNet Block Component**
- `Resnet1DBlock`: Custom layer implementing residual connections
- Different configurations for encode/decode modes
- Normalization layers (InstanceNorm for encoder, BatchNorm for decoder)

**VAE Encoder**
- Input Layer: Shape = (1, 90001)
- Convolutional layers with increasing filter sizes
- ResNet blocks between convolutional layers
- Output: Latent space representation (mean and logvar)

**VAE Decoder**
- Input Layer: Latent space vector
- Transposed convolutional layers with decreasing filter sizes
- ResNet blocks between layers
- Output: Reconstructed audio waveform

**Generation Components**
- `inference()`: Generates multiple audio samples from test data
- Latent space sampling via reparameterization trick
- Audio visualization and playback

### Data Flow Diagrams (DFDs)

#### Bidirectional LSTM Flow
```
[MIDI Files] → [Note Extraction] → [Sequence Creation] → [Training Data]
                                                          ↓
[Seed Data]  → [Trained Model] → [Note Generation] → [MIDI Conversion] → [Sheet Music]
```

#### VAE Flow
```
[Audio Files] → [Waveform Processing] → [Training Data] → [VAE Training]
                                                          ↓
[Test Samples] → [Encoder] → [Latent Space] → [Decoder] → [Generated Audio]
                  ↘                ↓                        ↓
                   [Visualization] → [Animation] → [Audio Playback]
```

## 3. Implementation Documentation

### Source Code Documentation

#### music-generation-lstm.ipynb
This notebook implements the bidirectional LSTM approach for classical music generation.

**Key Components:**
- Data loading and preprocessing from MIDI files
- Note extraction and corpus creation
- Rare note filtering and sequence preparation
- Bidirectional LSTM model definition and training
- Music generation function with visualization capabilities

**Model Architecture:**
```python
model = Sequential()
model.add(Bidirectional(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))
```

#### music-vae-jazz.ipynb
This notebook implements a VAE approach for jazz music generation.

**Key Components:**
- Audio data loading and processing from GTZAN dataset
- ResNet block definition for convolutional operations
- VAE architecture with encoder and decoder components
- Custom loss functions for VAE training
- Generation and visualization of jazz audio outputs

**VAE Architecture Highlights:**
```python
# Encoder structure
self.encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,90001)),
    layers.Conv1D(64,1,2),
    Resnet1DBlock(64,1),
    # More layers...
    layers.Dense(latent_dim+latent_dim)
])

# Decoder structure
self.decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    layers.Reshape(target_shape=(1,latent_dim)),
    Resnet1DBlock(512,1,'decode'),
    # More layers...
    layers.Conv1DTranspose(90001,1,1),
])
```

#### music-vae-classical.ipynb
This notebook adapts the VAE approach for classical music generation.

**Key Components:**
- Similar architecture to jazz VAE but trained on classical music
- Same ResNet block and VAE components
- Training and generation of classical music audio outputs

### API Documentation

#### Data Processing APIs

**MIDI Processing Functions**
- `extract_notes(file)`: Extracts notes and chords from MIDI files
  - **Parameters**: `file` - List of MIDI objects
  - **Returns**: List of string representations of notes and chords

- `chords_n_notes(Snippet)`: Converts note strings to MIDI objects
  - **Parameters**: `Snippet` - List of note strings
  - **Returns**: Stream object containing notes and chords

**Audio Processing Functions**
- `load(file_)`: Loads audio file as numpy array
  - **Parameters**: `file_` - Path to audio file
  - **Returns**: Numpy array of shape (1, 90001)

- `DatasetLoader(class_)`: Creates training and testing sets from audio files
  - **Parameters**: `class_` - Genre folder name
  - **Returns**: Two lists of file paths for training and testing

#### Model APIs

**LSTM Generation API**
- `Melody_Generator(Note_Count)`: Generates musical sequence
  - **Parameters**: `Note_Count` - Number of notes to generate
  - **Returns**: Tuple of (generated notes, MIDI melody object)

**VAE APIs**
- `CVAE.encode(x)`: Encodes input audio to latent space
  - **Parameters**: `x` - Input audio tensor
  - **Returns**: Mean and logvar tensors
  
- `CVAE.decode(z, apply_sigmoid)`: Decodes from latent space
  - **Parameters**: `z` - Latent vector, `apply_sigmoid` - Boolean
  - **Returns**: Reconstructed audio tensor

- `CVAE.sample(eps)`: Generates samples from latent space
  - **Parameters**: `eps` - Optional noise tensor
  - **Returns**: Generated audio samples

- `inference(test_dataset, model)`: Generates multiple audio samples
  - **Parameters**: `test_dataset` - Dataset, `model` - Trained VAE
  - **Returns**: List of generated audio samples

### Version Control History

*This section would typically contain commit history and git logs. For documentation purposes, we'll outline a hypothetical development timeline.*

1. Initial project setup and repository creation
2. MIDI data processing implementation
3. Implementation of basic LSTM model for note generation
4. Enhancement to bidirectional LSTM architecture
5. Implementation of VAE architecture for audio generation
6. Jazz music generation module development
7. Classical music VAE adaptation
8. Visualization and playback features
9. Performance optimizations and parameter tuning
10. Documentation and code comments

## 4. Testing Documentation

### Test Plan and Strategy

#### Unit Testing
- Test individual functions for data processing
- Validate model components in isolation
- Verify correct input/output shapes for each layer

#### Integration Testing
- Test full data processing pipeline
- Validate model training workflow
- Test generation process end-to-end

#### Performance Testing
- Measure training time on standard hardware
- Benchmark generation speed for different output lengths
- Memory usage profiling during training and generation

#### User Acceptance Testing
- Subjective evaluation of musical quality
- Comparison between different generation approaches
- User feedback on interface and outputs

### Test Cases

#### LSTM Model Test Cases

1. **Data Processing Test**
   - **Input**: Set of MIDI files
   - **Expected**: Correctly extracted notes and chords
   - **Validation**: Count and format of extracted elements

2. **Model Architecture Test**
   - **Input**: Model definition
   - **Expected**: Correct layer structure
   - **Validation**: Model summary matches specification

3. **Training Progress Test**
   - **Input**: Training data
   - **Expected**: Decreasing loss over epochs
   - **Validation**: Loss curve shows learning

4. **Generation Test**
   - **Input**: Seed sequence
   - **Expected**: Coherent note sequence
   - **Validation**: Generated notes follow musical patterns

#### VAE Model Test Cases

1. **Encoder Test**
   - **Input**: Audio waveform
   - **Expected**: Mean and logvar in latent space
   - **Validation**: Output shape matches latent dimension

2. **Decoder Test**
   - **Input**: Latent vector
   - **Expected**: Reconstructed audio
   - **Validation**: Output shape matches input shape

3. **Loss Function Test**
   - **Input**: Original and reconstructed audio
   - **Expected**: Valid loss calculation
   - **Validation**: Loss decreases during training

4. **Generation Quality Test**
   - **Input**: Random latent vectors
   - **Expected**: Diverse yet coherent audio samples
   - **Validation**: Subjective listening evaluation

### Bug Reports and Test Summaries

*In a real project, this section would contain actual bug reports and test results. For this document, we'll provide a template and hypothetical examples.*

#### Bug Report Example

**Bug ID**: BUG-001  
**Title**: Rare note filtering causing corpus length inconsistency  
**Description**: When removing rare notes, the current approach may cause inconsistencies in the corpus length due to direct list modification during iteration.  
**Steps to Reproduce**:
1. Create a corpus with notes appearing fewer than 100 times
2. Run the rare note filtering code
3. Observe the corpus length before and after

**Expected Result**: All rare notes removed consistently  
**Actual Result**: Some rare notes may remain due to list modification during iteration  
**Severity**: Medium  
**Status**: Fixed - Replaced with a filter approach that creates a new list

#### Test Summary

**Test Suite**: LSTM Generation Tests  
**Date**: May 4, 2025  
**Results Summary**:
- Total Tests: 15
- Passed: 13
- Failed: 2
- Success Rate: 86.7%

**Failed Tests**:
1. Memory usage within limits (exceeded on long generation)
2. Musical coherence at high diversity settings

**Actions**:
- Optimize memory management in generation function
- Add additional constraints on diversity parameter

## 5. Deployment Documentation

### Deployment Guide

#### System Requirements
- Python 3.7 or higher
- TensorFlow 2.x
- CUDA support recommended for faster training
- Minimum 8GB RAM for training, 4GB for inference
- Storage: At least 5GB for datasets and models

#### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/username/MuseNetic.git
   cd MuseNetic
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv musenetic-env
   # On Windows
   musenetic-env\Scripts\activate
   # On Linux/Mac
   source musenetic-env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download datasets:
   ```bash
   python scripts/download_datasets.py
   ```

5. Verify installation:
   ```bash
   python scripts/verify_setup.py
   ```

#### Running the System
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open one of the notebook files:
   - music-generation-lstm.ipynb for classical MIDI generation
   - music-vae-jazz.ipynb for jazz audio generation
   - music-vae-classical.ipynb for classical audio generation

3. Follow the in-notebook instructions for execution

### Release Notes

#### Version 1.0.0 (Initial Release)
- Bidirectional LSTM model for classical music generation
- VAE models for jazz and classical audio generation
- Visualization tools for both approaches
- Basic export functionality for generated music

#### Features
- Classical music generation from MIDI training data
- Jazz and classical music generation from audio data
- Interactive visualization of generated music
- Audio playback of generated samples
- Export capabilities for further use

#### Known Limitations
- Training requires significant computational resources
- Generation quality depends on training data quality
- Limited control over musical structure
- No real-time generation capability

### Configuration Manuals

#### Data Configuration
- **MIDI Data Path**: Set in `filepath` variable in LSTM notebook
- **Audio Data Path**: Set in `BASE_PATH` variable in VAE notebooks
- **Output Directory**: Set paths for generated files as needed

#### Model Configuration

**LSTM Model Parameters**:
- Sequence Length: Modify `length` variable (default: 40)
- Batch Size: Set in `model.fit()` call (default: 256)
- Epochs: Set in `model.fit()` call (default: 200)
- Learning Rate: Set in Adamax optimizer (default: 0.005)

**VAE Model Parameters**:
- Latent Dimension: Set in `latent_dim` variable (default: 2)
- Batch Size: Set in `BATCH_SIZE` variable (default: 10)
- Epochs: Set in `epochs` variable (default: 20)
- Learning Rate: Set in Adam optimizer (default: 0.0003)

#### Generation Configuration

**LSTM Generation**:
- Note Count: Pass to `Melody_Generator()` function (example: 100)
- Diversity: Adjust value in softmax temperature calculation (default: 1.0)

**VAE Generation**:
- Sample Count: Set in test dataset size
- Custom Seeds: Modify random vector in inference function

## 6. Maintenance Documentation

### Change Logs

#### Version 1.0.1
- Fixed memory leak in LSTM generation process
- Improved VAE training stability
- Added additional visualization options
- Updated dependencies to latest versions

#### Version 1.0.2
- Enhanced rare note filtering algorithm
- Added export options for different audio formats
- Improved documentation and code comments
- Fixed visualization issues in Jupyter notebooks

#### Version 1.1.0
- Added parameter tuning capabilities
- Implemented model checkpointing for interrupted training
- Enhanced audio preprocessing for better quality
- Added comparative visualization between generation methods

### Issue Tracking Reports

*In a real project, this would contain actual issues from an issue tracking system. Below are example templates.*

#### Open Issues

**Issue #15: High Memory Usage During Training**
- **Type**: Performance
- **Priority**: Medium
- **Description**: VAE training consumes excessive memory on longer audio samples
- **Assigned To**: Data Team
- **Planned Resolution**: Implement batch processing for large audio files

**Issue #23: Generation Diversity Control**
- **Type**: Feature Enhancement
- **Priority**: High
- **Description**: Add more granular control over output diversity
- **Assigned To**: Model Team
- **Planned Resolution**: Implement adjustable temperature parameter

#### Resolved Issues

**Issue #7: MIDI Export Format Compatibility**
- **Resolution**: Added support for multiple MIDI format versions
- **Fixed In**: Version 1.0.2
- **Resolution Notes**: Implemented music21 MidiFile export options

**Issue #12: Training Progress Visualization**
- **Resolution**: Added real-time plotting of loss metrics
- **Fixed In**: Version 1.1.0
- **Resolution Notes**: Implemented Matplotlib animation for loss curves

### Updated User and Admin Manuals

#### Administration Guide Updates

**Model Management**
1. Managing trained models:
   - Models are saved in the `models/` directory
   - Use naming convention `{model_type}_{date}_{parameters}.h5`
   - Regular backups recommended for valuable trained models

2. System monitoring:
   - Check GPU memory usage during training
   - Monitor disk space for generated outputs
   - Log training metrics for comparison

**Dataset Management**
1. Adding new training data:
   - Place MIDI files in `input/classical-music-midi/` directory
   - Place audio files in appropriate genre folders in GTZAN dataset
   - Run preprocessing scripts to update data indexes

2. Data quality assurance:
   - Use provided scripts to validate MIDI file integrity
   - Check audio files for consistency in sampling rate and length
   - Remove corrupt files before training

## 7. User and Support Documentation

### User Manual

#### Getting Started
1. **System Overview**: MuseNetic generates music using AI through two approaches:
   - Note-by-note generation using bidirectional LSTM (classical music)
   - Audio waveform generation using VAE (jazz and classical music)

2. **First Steps**:
   - Open the desired notebook in Jupyter
   - Run cells sequentially to understand the workflow
   - Default parameters are set for best results

#### Using the LSTM Music Generator
1. Load and preprocess MIDI data
2. Train the model or load a pre-trained model
3. Generate music by specifying note count
4. Visualize the generated music as sheet notation
5. Export to MIDI or listen to the generated audio

#### Using the VAE Music Generator
1. Load and preprocess audio data
2. Train the VAE model or load a pre-trained model
3. Generate music samples from the latent space
4. Visualize waveforms of generated audio
5. Listen to multiple generated samples
6. Export selected samples as audio files

### Help Files and FAQs

#### Troubleshooting

**Q: The model training is very slow. How can I speed it up?**  
A: Consider using a machine with GPU support. Set `BATCH_SIZE` higher if memory allows. Reduce model complexity for faster training at the cost of some quality.

**Q: The generated music doesn't sound coherent. How can I improve it?**  
A: Try longer training periods. Use larger or more diverse training datasets. Adjust the diversity parameter in generation (lower values for more coherent but less creative outputs).

**Q: I'm getting memory errors during VAE training.**  
A: Reduce batch size. Process shorter audio segments. Use a machine with more RAM or enable swap space.

**Q: How do I create my own training dataset?**  
A: For LSTM, collect MIDI files and place them in the input directory. For VAE, prepare audio files of consistent length and format, organized by genre.

#### Common Tasks

**Exporting Generated Music**
- LSTM model: Use `Melody.write('midi','filename.mid')` to save as MIDI
- VAE model: Use provided audio export functions to save as WAV

**Customizing Generation**
- Adjust sequence length for different musical phrases
- Modify temperature parameter for diversity control
- Sample different regions of the latent space in VAE

### Training Materials

#### Tutorial: Getting Started with LSTM Music Generation

1. **Understanding the Data**:
   - MIDI files contain note sequences and timing information
   - Notes are extracted and encoded as sequences
   - Sequences are used to predict the next note

2. **Model Training Process**:
   - Data preprocessing and sequence creation
   - Model configuration and compilation
   - Training with loss monitoring
   - Model evaluation

3. **Music Generation**:
   - Seed selection for generation start
   - Note-by-note generation process
   - Converting generated notes to MIDI
   - Fine-tuning for better results

#### Tutorial: VAE Music Generation

1. **Understanding VAE Architecture**:
   - Encoder compresses audio to latent space
   - Latent space provides compact representation
   - Decoder reconstructs audio from latent vectors

2. **Training Process**:
   - Data loading and preprocessing
   - Defining loss functions (reconstruction + KL divergence)
   - Training loop execution
   - Visualization of training progress

3. **Exploring the Latent Space**:
   - Sampling random points for diverse generation
   - Interpolating between points for smooth transitions
   - Finding areas with desirable musical qualities

#### Workshop Materials: AI Music Composition

1. **Comparing Generation Approaches**:
   - Sequence-based vs. waveform-based generation
   - Trade-offs in quality and control
   - Appropriate use cases for each approach

2. **Extending the Models**:
   - Adding conditioning for style transfer
   - Incorporating musical constraints
   - Creating hybrid approaches

3. **Creative Applications**:
   - Using generated music in compositions
   - Combining with traditional composition techniques
   - Ethical considerations in AI music generation