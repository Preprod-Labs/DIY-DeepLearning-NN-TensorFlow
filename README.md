# Keras & Streamlit Lab

This project is a deep learning application that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The application features a Streamlit-based user interface for training the model, visualizing the training process, and making predictions.

---
To learn more about the project, click [here](prompts.md)


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Model Architecture](#model-architecture)

## Features

- User-friendly Streamlit interface for model training and prediction
- Customizable training parameters (learning rate, epochs, batch size)
- Real-time training progress visualization
- Interactive charts for loss and accuracy metrics
- Image prediction functionality

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Preprod-Labs/DIY-DeepLearning-NN-TensorFlow
   ```

2. Install `conda` for dependency management:
   ```
   pip install conda
   ```

3. Create a virtual environment and activate it:
   ```
   conda create -n myenv -y
   conda activate myenv
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset:
   - Place your training images in `dataset/training_set/training_set`
   - Place your testing images in `dataset/test_set/test_set`
   - Ensure each set has subdirectories for `cats` and `dogs`

2. Run the Streamlit app:
   ```
   streamlit run KerasTrainer.py
   ```

3. Use the Streamlit interface to:
   - Set data paths
   - Configure training parameters
   - Start the training process
   - Visualize training progress and results
   - Make predictions on new images

## Code Structure

The main script `KerasTrainer.py` is organized into several sections:

1. **Imports and Constants**: Required libraries and global variables.
2. **Data Loading and Preprocessing**: Functions to load and prepare image data.
3. **Model Creation**: Definition of the CNN architecture.
4. **Visualization Functions**: Methods for plotting training history and predictions.
5. **StreamlitCallback**: Custom Keras callback for updating the Streamlit UI during training.
6. **Main Function**: Streamlit UI setup and main execution flow.

## Model Architecture

For detailed information about the model architecture, please refer to the [ModelArchitecture.md](ModelArchitecture.md) file.