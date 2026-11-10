# 🩺 Heart Disease Prediction System

An end-to-end machine learning pipeline and interactive diagnostic tool built to predict cardiovascular disease risk using clinical patient parameters. The project covers the full data science lifecycle—from exploratory data analysis and model benchmarking to hyperparameter tuning, model persistence, and an interactive testing dashboard.

---

## 📋 Project Overview
* **Dataset:** Cleveland Heart Disease Dataset (303 records, 13 clinical and demographic features).
* **Target Variable:** Binary classification (`0` = Healthy / Absence of Heart Disease, `1` = Presence of Heart Disease).
* **Best Model:** Tuned K-Nearest Neighbors (KNN) Classifier (achieving **83.61%** test accuracy).
* **Interface:** Interactive `ipywidgets` dashboard designed for real-time inference within Google Colab.

---

## 📊 Features & Clinical Attributes
The model evaluates 13 key physiological and diagnostic indicators:
* **Age:** Patient age in years
* **Gender:** Biological sex (0 = Female, 1 = Male)
* **Chest Pain Type:** Type of chest pain experienced (0–3 scale)
* **Resting Blood Pressure:** Admission blood pressure (mm Hg)
* **Cholesterol:** Serum cholesterol measurement (mg/dl)
* **Fasting Blood Sugar:** Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)
* **Resting ECG:** Resting electrocardiographic results (0–2 scale)
* **Max Heart Rate:** Maximum heart rate achieved during stress testing
* **Exercise Induced Angina:** Chest pain induced by physical exertion (0 = No, 1 = Yes)
* **ST Depression:** Exercise-induced ST depression relative to rest (Oldpeak)
* **ST Slope:** Slope of the peak exercise ST segment (0–2 scale)
* **Major Vessels Colored:** Number of major vessels colored by fluoroscopy (0–4)
* **Thalassemia:** Blood flow disorder classification (0–3 scale)

---

## 🤖 Model Benchmarking & Tuning
Four core classification algorithms were trained and evaluated on an 80-20 train-test split with feature scaling (`StandardScaler`):
1. **K-Nearest Neighbors (KNN):** Top baseline performer (~90.16% initial accuracy).
2. **Support Vector Classifier (SVC):** 86.89% accuracy.
3. **Logistic Regression:** 85.25% accuracy.
4. **Random Forest Classifier:** 83.61% accuracy.

**Hyperparameter Optimization:** 
Using `GridSearchCV`, the KNN model was optimized across neighbor sizes, weights, and distance metrics. The optimal configuration (`manhattan` distance, `n_neighbors=5`, `uniform` weights) yielded a cross-validation score of **83.84%** and a generalized test accuracy of **83.61%**.

---

## 📁 Repository Structure
```text
├── heart_disease_model.pkl    # Serialized tuned KNN model
├── scaler.pkl                 # Serialized StandardScaler object
├── notebook.ipynb             # Full end-to-end Colab notebook (EDA, training, UI)
└── README.md                  # Project documentation
