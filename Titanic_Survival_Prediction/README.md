# Titanic Survival Prediction: A CRISP-DM Implementation

This repository contains a robust end-to-end machine learning pipeline for the Titanic: Machine Learning from Disaster competition. The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to move from raw data to a production-ready classifier.

## 📊 Executive Summary
The objective was to build a predictive model to determine passenger survival with a focus on generalization and pipeline integrity. By implementing an automated Scikit-Learn pipeline, I achieved a **83.43% Cross-Validation accuracy**, effectively handling missing data and high-cardinality categorical variables.

## 🛠️ Technical Stack
* **Language:** Python 3.12
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Framework:** CRISP-DM
* **Core Algorithm:** Random Forest Classifier

## 🏗️ Pipeline Architecture
To prevent data leakage and ensure reproducibility, I developed a modular `ColumnTransformer` integrated into a `Pipeline`:
* **Numerical Branch:** `SimpleImputer` (Median) → `StandardScaler`
* **Categorical Branch:** `SimpleImputer` (Most Frequent) → `OneHotEncoder` (handling unknown labels)
* **Validation Strategy:** `StratifiedShuffleSplit` (20% hold-out) and 5-Fold Cross-Validation.

## 📈 Optimization & Results
* **Baseline Model:** SGDClassifier (~80.45% accuracy).
* **Optimized Model:** Random Forest via `GridSearchCV`.
* **Optimal Hyperparameters:** `max_depth: 15`, `n_estimators: 400`.
* **Diagnostic Tools:** Analyzed Learning Curves to mitigate high variance (overfitting) and monitored Confusion Matrices to balance precision/recall.

## 📁 File Structure
* `main.ipynb`: Comprehensive development environment (EDA, Preprocessing, Modeling).
* `titanic_survival_model.pkl`: Serialized production-ready pipeline.
* `titanic_submission.csv`: Final predictions for the Kaggle test set.