# Decision Tree for Gender Prediction

This project implements a Decision Tree Classifier from scratch (without `sklearn`) to predict gender based on height, weight, and age.

## 📌 Features
- Implements Decision Tree using **Information Gain**.
- Trained on a dataset containing **height, weight, and age**.
- Evaluates accuracy at different depths to analyze overfitting.
- Uses only `numpy` for calculations (no `sklearn`).

## 📊 Results
- Accuracy at different tree depths:

| Depth | Train Accuracy | Test Accuracy |
|-------|---------------|--------------|
| 1     | 40%           | 40%          |
| 2     | 50%           | 50%          |
| 3     | 60%           | 60%          |
| 4     | 70%           | 70%          |
| 5     | 80%           | 80%          |

## 📌 Observations:

- As depth increases, training accuracy improves.
- However, overfitting can occur at higher depths.

## 📝 Methodology
### 1. Data Processing
- The dataset contains features for height, weight, and age.
- Labels are converted to binary values (M → 0, W → 1).

### 2. Decision Tree Implementation
- The tree is built recursively using Information Gain.
- Each node splits on the best threshold.
### - Stopping Conditions:
-- Minimum samples split = 2
-- Max depth = 5

### 3. Model Training & Evaluation
- Trained on 50 samples.
- Tested on 70 samples.
- Accuracy comparison between training & test sets.
