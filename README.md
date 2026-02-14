## Machine Learning Assignment 2 – Classification and Streamlit App

This repository contains the work for **Machine Learning – Assignment 2**.

The main goals are:
- Implement multiple classification models on a real dataset.
- Evaluate the models using standard metrics.
- Build a simple **Streamlit** web app to demonstrate the models.
- Deploy the app on **Streamlit Community Cloud**.

Everything is kept intentionally simple so it is easy to understand and explain.

---

## 1. Problem Statement

Use a real-world **classification dataset** to:
- Train and compare different machine learning classification models.
- Evaluate them using various performance metrics.
- Expose the models through an interactive Streamlit web application.

The app should allow a user to:
- See the dataset and its basic statistics.
- Compare model performance.
- Upload a CSV file (test data) and get predictions from a selected model.

---

## 2. Dataset Description

**Dataset used:** Heart Disease dataset  
**Source:** CSV file `data/heart_disease_dataset.csv` (included in this repository)

- **Type:** Binary classification  
- **Classes (target column):**  
  - 0 – No Heart Disease  
  - 1 – Heart Disease  
- **Number of instances (rows):** 1,025  
- **Number of features (columns):** 13  
  - `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg` (resting ECG), `thalach` (max heart rate), `exang` (exercise induced angina), `oldpeak`, `slope`, `ca`, `thal`

You can see the first few rows and class distribution directly inside the Streamlit app.

### 2.1 Simple Preprocessing Steps

Before training the models, the code performs a few basic cleaning steps:

- **Drop duplicate rows** (if any) from the dataset.  
- **Fill missing numeric values** with the **median** of each column.  
- **Scale all feature columns** (StandardScaler) inside each model’s Pipeline so that:
  - Features are on a similar scale.
  - Models like Logistic Regression and kNN can learn better.
  - The same scaling is automatically applied later when making predictions in the Streamlit app.

---

## 3. Models Used and Evaluation Metrics

The following 6 machine learning models are implemented on the **same dataset**:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### 3.1 Comparison Table of Evaluation Metrics

For each model, the following metrics are calculated (on the test set):
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

After you run the training (see instructions below), you can fill this table with your actual numbers in your assignment PDF:

| ML Model Name           | Accuracy | AUC  | Precision | Recall | F1  | MCC  |
|-------------------------|---------:|-----:|----------:|-------:|----:|-----:|
| Logistic Regression     |   value  |value|    value  | value  |value|value |
| Decision Tree           |   value  |value|    value  | value  |value|value |
| kNN                     |   value  |value|    value  | value  |value|value |
| Naive Bayes             |   value  |value|    value  | value  |value|value |
| Random Forest (Ensemble)|   value  |value|    value  | value  |value|value |
| XGBoost (Ensemble)      |   value  |value|    value  | value  |value|value |

> Tip: When you run `python -m model.train_models` the metrics table will be printed in the terminal. You can copy those values into this table for your PDF.

### 3.2 Observations about Model Performance

Here, you should **write your own observations** about how each model behaves on this dataset.  
Some example points you might consider (write them yourself, in your own words):
- Which model has the highest accuracy?
- Which model has the best precision/recall for malignant cases?
- Is there any model that clearly overfits or underfits?
- Are ensemble models performing better than single models?

You can organize your observations in a table like this in your submission PDF:

| ML Model Name           | Observation about model performance                 |
|-------------------------|-----------------------------------------------------|
| Logistic Regression     | your own observation                                |
| Decision Tree           | your own observation                                |
| kNN                     | your own observation                                |
| Naive Bayes             | your own observation                                |
| Random Forest (Ensemble)| your own observation                                |
| XGBoost (Ensemble)      | your own observation                                |

---

## 4. Project Structure

The repository follows the structure suggested in the assignment:

```text
project-folder/
│-- app.py                    # Main Streamlit app
│-- requirements.txt          # Python dependencies
│-- README.md                 # This file
│-- data/
│   └── heart_disease_dataset.csv   # Heart disease dataset (13 features + target)
│-- model/
    ├── train_models.py       # Code to load data, train, evaluate and save models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    └── xgboost_ensemble.pkl
```

> Note: The `.pkl` files are created automatically when you run  
> `python -m model.train_models`. They are the **saved versions of the trained models**
> and can be used later if you want to load models from disk instead of retraining.

You can add more files (e.g., separate `.py` or `.ipynb` notebooks for each model) if needed, but this minimal structure is enough to complete the assignment.

---

## 5. How to Run Everything (Step-by-Step)

These steps assume you are running on **BITS Virtual Lab** (or any machine with Python 3 installed).

### 5.1. Create and Activate a Virtual Environment (recommended)

```bash
cd ml-assignment-2

# Create virtual environment (only once)
python -m venv .venv

# Activate on macOS / Linux
source .venv/bin/activate

# On Windows (PowerShell)
# .venv\\Scripts\\Activate.ps1
```

### 5.2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5.3. Run the Training Script and See Metrics

This will:
- Load the dataset
- Split into train and test
- Train all 6 models
- Print the comparison table and classification reports

```bash
python -m model.train_models
```

In the terminal, you will see:
- Dataset shape (instances and features)
- A table with Accuracy, AUC, Precision, Recall, F1 and MCC for each model
- Classification report for each model

You can use these printed values to fill in the tables for your PDF submission.

### 5.4. Run the Streamlit App Locally

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

In the app you can:
- See dataset overview (tab: **Dataset Overview**)
- See model comparison, confusion matrix, and classification report (tab: **Model Comparison**)
- Upload a CSV with the same feature columns and get predictions (tab: **Predictions with Uploaded Data**)

> For the **Bits Virtual Lab screenshot**, run the app on the lab environment, open it in the browser there, and take a screenshot showing that it is running.

---

## 6. Deployment on Streamlit Community Cloud

1. **Push this repository to GitHub.**  
   Make sure that at least these files are present on GitHub:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/heart_disease_dataset.csv`
   - `model/train_models.py`

2. Go to `https://streamlit.io/cloud` in your browser.

3. Sign in with your **GitHub account**.

4. Click **“New app”**.

5. Select your repository:
   - Choose the GitHub repo you created for this assignment.
   - Choose the correct branch (usually `main`).
   - Set the main file to `app.py`.

6. Click **Deploy**.

7. Wait for a few minutes. Streamlit will:
   - Install dependencies from `requirements.txt`.
   - Run `app.py`.
   - Give you a **public URL** for your app.

8. Open the URL and check that:
   - The app loads without errors.
   - You can see the dataset.
   - You can select models, see metrics and confusion matrix.
   - You can upload a CSV and get predictions.

This **public URL** is what you will paste into your assignment PDF as the **Live Streamlit App Link**.

---

## 7. Final Submission Checklist (for you)

Before submitting your assignment PDF, check:

- **GitHub repository link works** and contains:
  - `app.py`
  - `requirements.txt`
  - `README.md`
  - `data/heart_disease_dataset.csv`
  - `model/train_models.py`
- **Streamlit app link works** and opens the UI.
- **App features**:
  - Dataset overview
  - Model selection dropdown
  - Display of evaluation metrics
  - Confusion matrix or classification report
  - CSV upload and predictions
- **README content** is copied into your PDF and updated with:
  - Actual metric values in the comparison table
  - Your own observations for each model
- **BITS Lab screenshot** (Streamlit app running there) is added to the PDF.

> Important: Write the final observations and explanatory text in your **own words** to avoid plagiarism issues.
