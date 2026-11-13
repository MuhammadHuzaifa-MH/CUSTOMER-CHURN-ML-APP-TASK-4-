# 🛡️ ChurnGuard AI — Enterprise Customer Churn Prediction System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Devixo Solutions — AI/ML Internship Program | Task 04: End-to-End Machine Learning Application with Deployment*

</div>

---

## 👨‍💻 Intern Details
* **Name:** Muhammad Huzaifa
* **Submission Date:** 13 Nov 2026

---

## 📋 1. Introduction & Objective
Customer churn represents the percentage of customers who stop subscribing to a company's product or service within a given timeframe. Retaining existing customers is significantly more cost-effective than acquiring new ones. 

The objective of **Task 04** is to build a complete, production-ready Machine Learning web application that processes real-world customer data, trains and tunes multiple classification algorithms, serializes the winning pipeline, and deploys an interactive dashboard for real-time and batch churn predictions.

---

## ✨ 2. Key Features & Capabilities
1. **🔐 Secure Authentication Portal:** Password-protected enterprise login screen before accessing the dashboard.
2. **🔮 Real-Time Single Prediction Dashboard:** Interactive sliders and account inputs returning immediate churn risk classification, probability percentage, and dynamic visual progress bars.
3. **📡 Live Telemetry Stream:** Real-time simulated tracking graph displaying live API inference requests, system latency, and fluctuating risk telemetry.
4. **📁 Batch CSV Processing:** Upload bulk customer CSV datasets to predict churn across hundreds of records simultaneously and download results as a CSV.
5. **📈 Exploratory Data Analysis (EDA):** Visual distribution charts and boxplots illustrating spending patterns and churn behavior.
6. **🏆 Model Benchmarks:** Full comparative summary of all trained models before and after `GridSearchCV` hyperparameter tuning.

---

## 🛠️ 3. Technologies & Libraries Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Tuning:** Scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GridSearchCV`, `ColumnTransformer`, `Pipeline`)
* **Model Serialization:** Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend & Deployment:** Streamlit
* **Version Control:** Git & GitHub

---

## ⚙️ 4. Data Processing & Feature Engineering Pipeline
* **Data Cleaning:** Removed missing values and duplicate rows; dropped non-predictive identifier columns (`CustomerID`).
* **Outlier Capping:** Applied the Interquartile Range (IQR) method to cap extreme values across numerical columns.
* **Feature Engineering:** Created domain-specific interaction features:
  * `Spend_Per_Tenure = Total Spend / (Tenure + 1)`
  * `Support_Delay_Ratio = Support Calls / (Payment Delay + 1)`
* **Preprocessing:** Numerical columns imputed with median and scaled via `StandardScaler`; categorical columns imputed with most frequent values and encoded via `OneHotEncoder`.

---

## 📊 5. Model Development & Evaluation Results
Four distinct classification algorithms were trained and compared using stratified splits, followed by `GridSearchCV` hyperparameter optimization:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 89.37% | 88.50% | 91.20% | 90.47% | 0.9582 |
| **Decision Tree** | 99.93% | 99.90% | 99.95% | 99.94% | 0.9993 |
| **Random Forest** | 99.42% | 99.20% | 99.70% | 99.49% | 1.0000 |
| **Gradient Boosting (Tuned)** | **99.68%** | **99.50%** | **99.85%** | **99.72%** | **0.9997** |

---

## 📂 6. Repository Structure
```text
Customer-Churn-App/
│
├── churn.csv.                        # Real-world customer dataset
├── train_model.py                    # Data preprocessing, training, tuning & Joblib saving
├── churn_model.pkl                   # Serialized production ML pipeline
├── app.py                            # Interactive Streamlit web application
├── requirements.txt                  # Required Python libraries
├── Task_04_Customer_Churn_Report.md  # Formal project report (Intro, Objectives, Conclusion)
└── README.md                         # Project documentation
