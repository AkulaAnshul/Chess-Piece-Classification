# Chess Piece Classification Using Classical Machine Learning

This project classifies individual chess piece images using classical ML algorithms (SVM/MLP) and evaluates performance using confusion matrix, F1, precision, and PR curves.

## Features

- Cropped chess piece image classification (12 classes: 6 types × 2 colors)
- Feature extraction using Histogram of Oriented Gradients (HOG)
- Model training with SVM/MLP
- Standard evaluation metrics and visualizations

## Getting Started

1. Clone the repository.
2. Place your cropped chess piece images in `data/train` and `data/test` (see structure below).
3. Install dependencies:  
4. Run the notebook or scripts in `src/`.

## Folder Structure

- `data/`: Training and testing images
- `notebooks/`: Main Jupyter notebook
- `src/`: Python scripts for each pipeline stage
- `results/`: Output plots and metrics

## Results

Evaluation metrics and sample outputs are saved in `results/`.
