# Neural Network for Binary Classification

This project implements a binary image classifier using **PyTorch** and **Torchvision**. The dataset comes from **WorldQuant University**’s public computer vision resources.

---

##  Features

- Loads and preprocesses images (resize, normalization, RGB conversion).
- Custom NN model built with PyTorch.
- Uses `DataLoader` for batching and shuffling.
- Model training with cross-entropy loss and Adam optimizer.
- Performance evaluation with confusion matrix and accuracy metrics.
- Visualization of dataset distribution and sample images.

---

##  Project Structure

- `WQU.ipynb` — Main notebook containing full code (data prep → model → training → evaluation).  
- `requirements.txt` — List of dependencies.  
- `data_p1/` — Dataset folder (binary classification with hog vs blank).

---

##  Installation

```bash
git clone https://github.com/qasimmansoori/Neural-Network-Binary-classification.git
cd Neural-Network-Binary-classification
pip install -r requirements.txt
