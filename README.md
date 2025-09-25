# 🩺 Diabetes Prediction using Machine Learning

## 📊 Overview

This project leverages machine learning to predict the likelihood of diabetes in individuals based on various health parameters. The model achieves high accuracy by employing advanced techniques such as SMOTE for handling class imbalance and utilizing a robust pipeline for preprocessing and prediction.

## 🔬 Dataset

- **Source**: [Diabetes Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/your-dataset-link)
- **Features**:
  - Gender
  - Age
  - Hypertension
  - Heart Disease
  - Smoking History
  - BMI
  - HbA1c Level
  - Blood Glucose Level

## 🎯 Project Highlights

- **Accuracy**: Achieved ~97.2% accuracy using a well-processed dataset.
- **Techniques**:
  - Applied SMOTE to handle class imbalance.
  - Feature engineering and exploratory data analysis (EDA) included.
  - Clean and interactive Streamlit web interface.
  - Deployment-ready pipeline using `joblib`.
- **File Structure**:

.
├── data/
│   └── diabetes_prediction_dataset.csv
├── Deployment/
│   ├── model.py
│   └── diabetes_prediction_pipeline_V2.pkl
├── Notebook/
│   ├── Diabetes_prediction.ipynb
│   └── diabetes_prediction_pipeline_V2.pkl
└── README.md


## 🧪 Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, numpy, seaborn, matplotlib, joblib, xgboost
- **Web App**: Streamlit

## 🚀 How to Run the Project

1. **Clone the Repository**:

 ```bash
 git clone https://github.com/prince0420-bit/Diabetes-prediction-
 cd Diabetes-prediction-

streamlit run Deployment_Code.py
<img width="1920" height="1020" alt="Screenshot 2025-09-25 143643" src="https://github.com/user-attachments/assets/5fbfc2fa-dbe5-4993-8124-25da81c294f4" />
