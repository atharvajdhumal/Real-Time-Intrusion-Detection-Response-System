# Real-Time-Intrusion-Detection-Response-System
Enhancing Real Time Intrusion Detection &amp; Response Systems using AI/ML

This project presents a comprehensive pipeline for analyzing and building a machine learning-based intrusion detection system using the UNSW-NB15 dataset. It includes advanced Exploratory Data Analysis (EDA), feature engineering, model training, real-time prediction, and explainability using SHAP.

📁 Dataset
UNSW-NB15 (in .parquet format)

Contains both normal and malicious traffic samples labeled with binary class (0 = Normal, 1 = Attack) and detailed attack categories.

🚦 Key Features
1. 🧠 Advanced Exploratory Data Analysis (EDA)
Visualization of attack categories, protocol distributions, and traffic behavior.

Boxplots, KDE plots, and value counts for feature insights.

Risk mapping based on attack severity (Low / Medium / High).

2. 🔧 Preprocessing & Feature Engineering
Label encoding of categorical features (proto, service, state).

Binary and multi-class target options.

Risk level encoding for downstream threat response.

3. 🤖 Model Training & Evaluation
Trained three classifiers:

Random Forest

Logistic Regression

XGBoost

Metrics evaluated:

Accuracy, Precision, Recall, F1 Score, ROC AUC

Confusion matrices for each model

4. ⚡ Real-Time Prediction Engine
Predicts incoming traffic using trained models.

Outputs:

Prediction label (Normal / Attack)

Confidence score

Suggested action (Allow, Alert + Monitor, Alert + Block)

5. 🧠 Explainability with SHAP
Top 3 influential features behind real-time decisions.

SHAP visualizations of model feature importances.

📦 Requirements
pandas, numpy, matplotlib, seaborn

scikit-learn, xgboost, shap

🛠️ How to Use
Load UNSW_NB15_training-set.parquet and UNSW_NB15_testing-set.parquet

Run the script to perform EDA, training, and evaluation

Use the predict_realtime_traffic() or predict_realtime_with_shap() for real-time predictions

📌 Project Highlights
Suitable for cybersecurity research, SIEM integration, or machine learning security projects.

Built with explainable AI principles.

Ready for extension to multi-class classification and cloud-based deployment.
