# Diabetes Health Representation Using Stacked Autoencoder

## Project Overview
This project uses a Stacked Autoencoder to learn compact representations of diabetes health indicators. The encoded features are then used for classification and compared with a baseline machine learning model.

## Objectives
- Analyze the CDC Diabetes Health Indicators dataset.
- Preprocess health and lifestyle data.
- Learn compact feature representations using a Stacked Autoencoder.
- Train the model using Mini-Batch Gradient Descent.
- Compare reconstruction quality and classification performance.
- Evaluate the results using F1-score and a confusion matrix.

## Dataset
**Dataset:** CDC Diabetes Health Indicators  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

Please check the dataset version and target labels before reporting the number of classes. Do not upload the dataset unless its licensing terms permit redistribution.

## Technologies Used
- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Google Colab

## Methodology
1. Load and inspect the dataset.
2. Preprocess the data and split it into training, validation, and test sets.
3. Scale the input features without data leakage.
4. Build a Stacked Autoencoder.
5. Train using Mini-Batch Gradient Descent.
6. Extract encoded features.
7. Train a Logistic Regression classifier.
8. Compare with a baseline model.
9. Evaluate reconstruction loss, F1-score, and the confusion matrix.

## Model Architecture
The initial model uses encoder layers of 64 and 32 neurons, followed by a 16-dimensional latent representation. The decoder reconstructs the original input features.

## Evaluation
The project evaluates:
- Training and validation reconstruction loss
- Latent dimension comparisons
- Classification accuracy and F1-score
- Confusion matrix
- Baseline versus encoded-feature classification

Results should be added after successfully running and validating the notebook.

## Repository Structure
- `diabetes_stacked_autoencoder.ipynb` — notebook containing the implementation
- `README.md` — project documentation
- `requirements.txt` — required Python libraries

## Disclaimer
This is an academic machine-learning project and is not intended for clinical diagnosis.

## Results and Evaluation

The model is evaluated using the following measurements:

- Training and validation reconstruction loss
- Latent dimension comparison (8, 16, and 32)
- Classification accuracy
- Macro F1-score and weighted F1-score
- Classification report and confusion matrix

The results will be documented after successfully executing the notebook and verifying the outputs.

## How to Run

1. Open the notebook in Google Colab.
2. Install the required libraries listed in `requirements.txt`.
3. Load the CDC Diabetes Health Indicators dataset.
4. Execute the notebook cells in order.
5. Review the model evaluation metrics and plots.

## Disclaimer

This project is for academic and educational purposes only. It is not a medical diagnostic system.
