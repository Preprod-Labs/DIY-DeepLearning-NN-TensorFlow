# Keras & Streamlit Lab

This project is a deep learning application that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The application features a Streamlit-based user interface for training the model, visualizing the training process, and making predictions.

---
To Learn more about the project using your favorite LLM, click [Here](LearnWithPrompts.md)


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
   ```bash
   git clone https://github.com/Preprod-Labs/DIY-DeepLearning-NN-TensorFlow
   ```

2. Create a virtual environment and activate it. If using Conda:
   ```
   conda create -n *env_name* python==3.12.0 -y
   conda activate *env_name*
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset ( download cats and dogs dataset from your preffered sources such as Kaggle.com ):
   - Place your training images in `dataset/training_set/training_set`
   - Place your testing images in `dataset/test_set/test_set`
   - Ensure each set has subdirectories for `cats` and `dogs`

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Use the Streamlit interface to:
   - Set data paths
   - Configure training parameters
   - Start the training process
   - Visualize training progress and results
   - Make predictions on new images

## Code Structure

The main script `app.py` is organized into several sections:
(now kept in pyfiles directory)

1. **Imports and Constants**: Required libraries and global variables.
2. **Data Loading and Preprocessing**: Functions to load and prepare image data.
3. **Model Creation**: Definition of the CNN architecture.
4. **Visualization Functions**: Methods for plotting training history and predictions.
5. **StreamlitCallback**: Custom Keras callback for updating the Streamlit UI during training.
6. **Main Function**: Streamlit UI setup and main execution flow.

## Model Architecture

For detailed information about the model architecture, please refer to the [ModelArchitecture.md](ModelArchitecture.md) file.