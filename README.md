# ML Assignment 2 – Handwritten Digit Classification (5 models + Streamlit app)

**BITS Pilani WILP · M.Tech (AIML) · Machine Learning: Assignment 2**

**Student:** Himanshu Sharma (2026AK05017)

## 1. Problem statement
Classify 8x8 grayscale images of handwritten digits into the digits **0-9** (multi-class). Five classifiers are trained
on the same dataset, compared on six metrics, and exposed through an interactive Streamlit app that evaluates the
saved models on an uploaded test CSV (no retraining).

## 2. Dataset description
- Source: UCI Machine Learning Repository, *Optical Recognition of Handwritten Digits*
- Instances: 1,797 | Features: 64 (pixel intensities 0-16, columns `pixel_0_0` ... `pixel_7_7`) | Target: `digit` (10 balanced classes)
- Split: 80% train (1,437) / 20% test (360), stratified, `random_state=42`. The test split is saved as `test_data.csv`.

## 3. GitHub repository

- **GitHub repository:** https://github.com/bitshims2/mlassign2
- **Live Streamlit app:** https://mlassign2-2026hims.streamlit.app/

## 4. Models used and comparison table
Precision, Recall, F1 and AUC are weighted averages (AUC is one-vs-rest) because the problem is multi-class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |
| Decision Tree | 0.8222 | 0.9157 | 0.8230 | 0.8222 | 0.8209 | 0.8028 |
| KNN | 0.9639 | 0.9951 | 0.9648 | 0.9639 | 0.9636 | 0.9600 |
| Naive Bayes | 0.8111 | 0.9707 | 0.8480 | 0.8111 | 0.8151 | 0.7940 |
| Support Vector Machine | 0.9750 | 0.9995 | 0.9759 | 0.9750 | 0.9749 | 0.9723 |

## 5. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong (about 97% accuracy). Pixel intensities separate the digits almost linearly, and scaling helps convergence. |
| Decision Tree | Much weaker (about 82%). Axis-aligned splits on single pixels cannot capture stroke shapes, so similar digits are confused. |
| KNN | Strong (about 96%) because similar digits lie close together in pixel space; prediction is slower and depends on scaling and k. |
| Naive Bayes | Lowest accuracy (about 81%): neighbouring pixels are strongly correlated, which breaks the independence assumption. AUC stays high (0.97). |
| Support Vector Machine | Best on every metric (about 97.5% accuracy, MCC 0.972); the RBF kernel captures non-linear stroke patterns. |
| **Overall winner** | Support Vector Machine, narrowly ahead of Logistic Regression. |

## 6. Repository structure
```
streamlit_app.py     # Streamlit UI (loads saved models, never retrains)
train_models.ipynb   # trains the 5 models, saves model/*.pkl, model/meta.json and test_data.csv
requirements.txt
test_data.csv
model/               # logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, support_vector_machine.pkl
```

## 7. Run locally
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute train_models.ipynb   # or run all cells in Jupyter
streamlit run streamlit_app.py
```
