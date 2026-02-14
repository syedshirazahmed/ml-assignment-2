# ML Assignment 2
## Heart Disease Classification: Machine Learning Model Comparison

## Submitted By
- **Name:** Syed Shiraz Ahmed
- **BITS ID:** 2025AA05443

## 1. Github Repository Link
https://github.com/syedshirazahmed/ml-assignment-2

## 2. Live Streamlit App Link
https://ml-assignment-2-syedshirazahmed.streamlit.app/

## 3. Screenshot of BITS Lab

## 4. Github README Content
### a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict the presence of heart disease in patients based on clinical features. The problem is formulated as a binary classification task where:

- **Class 0:** No Heart Disease
- **Class 1:** Heart Disease Present

The project aims to:
1. Implement six different classification algorithms on the Heart Disease dataset
2. Evaluate and compare model performance using multiple metrics (Accuracy, AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient)
3. Identify the most effective model for heart disease prediction
4. Deploy an interactive web application using Streamlit for model demonstration and prediction

---

### b. Dataset Description

**Dataset:** Heart Disease Dataset  
**Source:** `https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset`

#### Dataset Characteristics

- **Type:** Binary Classification
- **Total Instances:** 1,025
- **Number of Features:** 13
- **Target Variable:** Presence (1) or absence (0) of heart disease
- **Class Distribution:** Balanced dataset with both classes well-represented

#### Features Description

The dataset contains the following 13 clinical features:

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age of the patient in years | Numeric |
| `sex` | Gender (1 = male, 0 = female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Categorical |
| `restecg` | Resting electrocardiographic results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise induced angina (1 = yes, 0 = no) | Categorical |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment (0-2) | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) | Numeric |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) | Categorical |

#### Further Notes

- **Critical Metric:** Recall is chosen as the critical metric because, as in the context of heart disease prediction, it is crucial to identify as many positive cases as possible. 
Missing a potential heart disease case (false negative) could have severe consequences.
- **Train/Test Split:** The Train/Test Split is 80/20.
- **Data Preprocessing:** EDA and Pre-processing of the data is also done. 

---

### c. Models Used

Six classification algorithms are implemented and evaluated:

#### 1. Logistic Regression
A linear model for binary classification using the logistic function. Simple, interpretable, and works well for linearly separable data.

#### 2. Decision Tree
A tree-based model that splits data based on feature thresholds. Captures non-linear relationships and provides interpretable decision rules.

#### 3. k-Nearest Neighbors (kNN)
An instance-based learning algorithm that classifies based on the majority class of k nearest neighbors. Non-parametric and effective for local patterns.

#### 4. Naive Bayes
A probabilistic classifier based on Bayes' theorem with strong independence assumptions. Fast and works well with limited training data.

#### 5. Random Forest (Ensemble)
An ensemble of decision trees using bagging. Reduces overfitting and improves generalization through majority voting.

#### 6. XGBoost (Ensemble)
An advanced gradient boosting algorithm. Builds trees sequentially to correct errors of previous trees, often achieving state-of-the-art performance.

---

### Model Performance Comparison

#### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8098 |	0.9298 |	0.7619 |	0.9143 |	0.8312 |	0.6309 |
| Decision Tree | 0.9854 |	0.9857 |	1 |	0.9714 |	0.9855 |	0.9712 |
| kNN | 0.8634 |	0.9629 |	0.8738 |	0.8571 |	0.8654 |	0.7269 |
| Naive Bayes | 0.8293 |	0.9043 |	0.807 |	0.8762 |	0.8402 |	0.6602 |
| Random Forest (Ensemble) | 1 |	1 |	1 |	1 |	1 |	1 |
| XGBoost (Ensemble) | 0.9707 |	0.9874 |	0.9714 |	0.9714 |	0.9714 |	0.9414 |

**Note:** Metrics are calculated on the test set (20% of the data).

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieves excellent recall (0.9143), successfully detecting 91.43% of heart disease cases - the most critical metric for patient safety. The model demonstrates solid baseline performance with 80.98% accuracy and strong AUC of 0.9298, indicating good discrimination ability across different thresholds. Precision is moderate at 0.7619, resulting in some false positives, but the high recall ensures only 8.57% of actual patients are missed. The F1 score of 0.8312 and MCC of 0.6309 confirm balanced performance. Its linear approach provides interpretability for understanding feature importance while prioritizing patient detection over false alarm reduction. |
| **Decision Tree** | **Outstanding performance** with exceptional recall (0.9714), missing only 2.86% of actual heart disease cases - one of the best for patient detection. Achieves 98.54% accuracy with perfect precision (1.0), meaning zero false positives alongside near-perfect sensitivity. The F1 score of 0.9855 reflects excellent balance, while the exceptionally high MCC (0.9712) confirms superior classification quality. AUC of 0.9857 demonstrates outstanding discrimination ability. This combination of high recall with perfect precision makes it ideal for clinical deployment, offering interpretable decision paths while minimizing both missed diagnoses and false alarms. |
| **kNN** | Shows good overall performance with 86.34% accuracy and strong AUC of 0.9629, indicating excellent ranking ability. However, recall (0.8571) is the lowest among competitive models, meaning 14.29% of heart disease cases might be missed - concerning for critical medical screening. The model achieves good precision (0.8738) with balanced F1 score of 0.8654 and solid MCC of 0.7269. While it effectively captures non-linear relationships through local similarity patterns, the recall limitation makes it less suitable than higher-sensitivity alternatives when minimizing missed diagnoses is paramount. |
| **Naive Bayes** | Delivers solid performance with 82.93% accuracy and good recall (0.8762), successfully identifying 87.62% of heart disease cases - valuable for medical screening. The model achieves good AUC of 0.9043 and balanced metrics with precision of 0.807, F1 of 0.8402, and MCC of 0.6602. Despite its strong independence assumption that likely doesn't hold for correlated medical features, the probabilistic approach performs reasonably well. While recall misses 12.38% of cases (moderate compared to top models), its simplicity and computational efficiency make it useful for rapid screening where acceptable sensitivity is needed with reasonable false positive rates. |
| **Random Forest (Ensemble)** | **Perfect performance across all metrics** - achieves 100% accuracy with flawless recall (1.0), precision (1.0), F1 (1.0), AUC (1.0), and MCC (1.0). The perfect recall means zero heart disease cases are missed, which is the most critical achievement for patient safety. Combined with perfect precision, the model eliminates both false negatives and false positives, correctly classifying every test instance. This exceptional result indicates the ensemble approach with proper hyperparameter tuning completely captures the underlying patterns. The gold standard for clinical deployment where complete disease detection with zero false alarms is paramount. |
| **XGBoost (Ensemble)** | **Exceptional performance** with 97.07% accuracy and consistently high metrics across the board. Most importantly, achieves excellent recall (0.9714), detecting 97.14% of heart disease cases and missing only 2.86% - tied with Decision Tree for second-best sensitivity. The model maintains perfect balance with equally excellent precision (0.9714), resulting in F1 of 0.9714. Outstanding AUC of 0.9874 demonstrates superior discrimination ability, while MCC of 0.9414 confirms excellent overall classification quality. The gradient boosting approach with regularization effectively learns complex patterns while preventing overfitting, making it an excellent choice for clinical screening where high recall and overall robustness are essential. |

---

### Key Findings

1. **Random Forest: Perfect Recall (1.0) - Zero Missed Cases:** Random Forest achieved perfect recall, detecting 100% of heart disease cases with zero false negatives. This is the most critical metric for patient safety, ensuring no patient goes undiagnosed. Combined with perfect precision, it represents the ideal model for clinical deployment.

2. **Top Tier Recall Performance:** Decision Tree and XGBoost both achieved excellent recall (0.9714), missing only 2.86% of cases. Logistic Regression follows with strong recall (0.9143), while Naive Bayes (0.8762) and kNN (0.8571) provide acceptable but lower sensitivity for critical medical screening.

3. **Recall as Primary Discriminator:** When ranked by recall - the most important metric for heart disease detection - the hierarchy is found to be as follows:
***Random Forest (1.0) > Decision Tree/XGBoost (0.9714) > Logistic Regression (0.9143) > Naive Bayes (0.8762) > kNN (0.8571).***
This ranking directly reflects clinical safety.

---