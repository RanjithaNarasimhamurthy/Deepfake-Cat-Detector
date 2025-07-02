
# Deepfake Cat Detector

A Jupyter Notebook–based deep learning project to detect deepfake images of cats using convolutional neural networks.

## Project Structure
- `Deepfake_Cat_Detector.ipynb` – Main notebook for training and evaluating the model.
- `requirements.txt` – List of Python dependencies.
- `models/` – Directory containing saved model weights.
- `data/` – Directory for datasets (e.g., real and deepfake cat images).

## Installation
```bash
git clone [<repository-url>](https://github.com/RanjithaNarasimhamurthy/)
cd Deepfake-Cat-Detector
pip install -r requirements.txt
```

## Usage
1. Launch the notebook:
   ```bash
   jupyter notebook Deepfake_Cat_Detector.ipynb
   ```
2. Follow the notebook steps to:
   - Load and preprocess the dataset.
   - Train the convolutional neural network.
   - Evaluate model performance and visualize results.

## Dependencies
- Python 3.8+
- TensorFlow (or PyTorch)
- scikit-learn
- numpy
- pandas
- matplotlib
- OpenCV

## Results
The model achieves high accuracy in distinguishing real cat images from deepfakes. Feel free to adjust hyperparameters and explore different architectures.

