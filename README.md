🩺 Diabetes Prediction using Machine Learning






A machine learning-powered web app for predicting the likelihood of diabetes based on patient health records.
Achieves an impressive 97.2% accuracy using a well-processed dataset with a Streamlit-based real-time prediction interface.

📊 Dataset

Source: Diabetes Prediction Dataset on Kaggle

Features:

🧑 Gender

🎂 Age

❤️ Hypertension

🫀 Heart Disease

🚬 Smoking History

⚖️ BMI

💉 HbA1c Level

🩸 Blood Glucose Level

🔍 Project Overview

Classifies whether a person is likely to have diabetes based on medical and lifestyle features using supervised machine learning.

🎯 Model Highlights

✅ Achieved ~97.2% accuracy

✅ Applied SMOTE to handle imbalanced data

✅ Extensive Feature Engineering & EDA

✅ Clean and interactive Streamlit web interface

✅ Deployment-ready joblib pipeline

📁 File Structure
.
├── data/
│   └── diabetes_prediction_dataset.csv
├── Deployment/
│   ├── Deployment_Code.py
│   └── diabetes_prediction_pipeline_V2.pkl
├── Notebook/
│   ├── Diabetes_prediction_model_smote.ipynb
│   ├── Diabetes_prediction_model.ipynb
│   ├── understanding_data.ipynb
│   └── diabetes_prediction_pipeline_V2.pkl   # duplicate model (optional)
└── README.md

🧪 Tech Stack

Language: Python

Libraries: scikit-learn, pandas, numpy, seaborn, matplotlib, joblib, xgboost

Web App: Streamlit

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/diabetes-prediction-ml.git
cd diabetes-prediction-ml



Add the trained model
Make sure diabetes_prediction_pipeline_V2.pkl is in the same folder as Deployment_Code.py.

Run the web app

streamlit run Deployment_Code.py

🧠 Model Training & Evaluation

Tested multiple classifiers: Logistic Regression, Random Forest, XGBoost

Selected the best model based on:

Precision

Recall

ROC-AUC

Accuracy

Handled class imbalance using SMOTE

Final model trained on full processed dataset and serialized using joblib

📷 App UI Preview

(Include screenshots of your Streamlit app here to make it visually appealing)

⚠️ Disclaimer

This project is for educational purposes only.
It is not a substitute for professional medical diagnosis. Always consult with a licensed healthcare provider.

🙌 Acknowledgements

Kaggle Dataset by iammustafatz

Inspired by the importance of early diabetes detection
