# -*- coding: utf-8 -*-

train_path = '/content/UNSW_NB15_training-set.parquet'
test_path = '/content/UNSW_NB15_testing-set.parquet'

# PART 1: ADVANCED EDA - UNSW-NB15
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot aesthetics
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Step 1: Load Data
train_path = '/content/UNSW_NB15_training-set.parquet'
test_path = '/content/UNSW_NB15_testing-set.parquet'

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Combine both for EDA
df_full = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

# Step 2: Quick Overview
print("Shape of Combined Data:", df_full.shape)
print("Columns:\n", df_full.columns.tolist())
print("Data Types:\n", df_full.dtypes.value_counts())
print("\nMissing Values:\n", df_full.isnull().sum().sort_values(ascending=False).head())

# Step 3: Target Columns
print("\nBinary Label Distribution:")
print(df_full['label'].value_counts())

print("\nAttack Category Distribution:")
print(df_full['attack_cat'].value_counts(dropna=False))

# Step 4: Attack Category by Binary Label
plt.figure(figsize=(12,6))
sns.countplot(data=df_full, y='attack_cat', hue='label', order=df_full['attack_cat'].value_counts().index)
plt.title("Attack Category Distribution (Split by Normal/Attack)")
plt.xlabel("Count")
plt.ylabel("Attack Category")
plt.legend(title='Label (0=Normal, 1=Attack)')
plt.tight_layout()
plt.show()

# Step 5: Top Protocols, Services, States by Attack
top_protos = df_full['proto'].value_counts().head(10)
top_services = df_full['service'].value_counts().head(10)
top_states = df_full['state'].value_counts().head(10)

print("\nTop Protocols:\n", top_protos)
print("\nTop Services:\n", top_services)
print("\nTop States:\n", top_states)

# Step 6: Plot top protocols vs label
plt.figure(figsize=(10,5))
sns.countplot(data=df_full[df_full['proto'].isin(top_protos.index)], x='proto', hue='label')
plt.title("Protocol Usage by Label")
plt.xlabel("Protocol")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

df_full.columns

# Step 7: Feature Summary
numerical_features = df_full.select_dtypes(include=['int64', 'float64'])
print("\nNumerical Feature Summary:\n")
print(numerical_features.describe().T[['mean', 'std', 'min', 'max']].round(2))

# Step 9: Boxplot (Log scale) for outlier detection
plt.figure(figsize=(14,6))
sns.boxplot(data=np.log1p(df_full[['dur', 'sbytes', 'dbytes', 'rate']]))
plt.title("Boxplot (Log Scale) of Key Numerical Features")
plt.tight_layout()
plt.show()

# Step 10: KDE for Traffic Features (normal vs attack)
for feature in ['dur', 'rate', 'sbytes', 'dbytes']:
    plt.figure(figsize=(10, 4))
    sns.kdeplot(data=df_full[df_full['label'] == 0], x=feature, label='Normal', fill=True)
    sns.kdeplot(data=df_full[df_full['label'] == 1], x=feature, label='Attack', fill=True)
    plt.title(f'Distribution of {feature} by Label')
    plt.legend()
    plt.tight_layout()
    plt.show()

from sklearn.preprocessing import LabelEncoder

# Step 1: Encode categorical variables (proto, service, state)
categorical_cols = ['proto', 'service', 'state']
encoded_df = df_full.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le
    print(f"{col} encoded — {len(le.classes_)} unique values")

# Step 2: Drop non-predictive identifiers (if any)
cols_to_drop = ['id', 'attack_cat'] if 'id' in encoded_df.columns else ['attack_cat']
encoded_df = encoded_df.drop(columns=cols_to_drop, errors='ignore')

# Step 3: Encode target variable (label: 0 = Normal, 1 = Attack)
y_binary = encoded_df['label']
X_full = encoded_df.drop(columns=['label'])

# Step 4: Optional — Encode attack categories for multi-class target
# Only do this if you plan multi-class later
if 'attack_cat' in df_full.columns:
    attack_le = LabelEncoder()
    df_full['attack_cat_encoded'] = attack_le.fit_transform(df_full['attack_cat'].astype(str))
    y_multiclass = df_full['attack_cat_encoded']
    print("\nEncoded attack_cat classes:\n", dict(zip(attack_le.classes_, attack_le.transform(attack_le.classes_))))

# Step 5: Optional — Risk Mapping (based on attack severity)
risk_map = {
    'Normal': 'Low',
    'Fuzzers': 'Medium', 'Reconnaissance': 'Medium',
    'Analysis': 'High', 'Backdoor': 'High', 'Shellcode': 'High', 'Worms': 'High',
    'DoS': 'High', 'Exploits': 'High', 'Generic': 'Medium'
}

# Fill risk for all rows (NaN for Normal automatically becomes 'Low')
df_full['risk'] = df_full['attack_cat'].map(risk_map).fillna('Low')

# Step 6: Binary Risk Encoding (for alerting systems)
df_full['risk_level'] = df_full['risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Step 7: Final data preview
print("\nFinal encoded shape:", X_full.shape)
print("\nSample rows with risk mapping:")
print(df_full[['attack_cat', 'risk', 'risk_level']].drop_duplicates())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Split back into train and test using original index (already labeled from parquet)
X = X_full.copy()
y = y_binary.copy()

# Ensure alignment
X['label'] = y.values
X['risk'] = df_full['risk'].values
X['risk_level'] = df_full['risk_level'].values

# Use original files to split
train_len = len(pd.read_parquet('/content/UNSW_NB15_training-set.parquet'))
X_train = X.iloc[:train_len].drop(columns=['label', 'risk', 'risk_level'])
X_test = X.iloc[train_len:].drop(columns=['label', 'risk', 'risk_level'])
y_train = y[:train_len]
y_test = y[train_len:]
risk_test = df_full['risk'][train_len:]
attack_type_test = df_full['attack_cat'][train_len:]

# Step 2: Train a Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Step 3: Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

# Step 4: Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 5: Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Binary Classification")
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prepare comparison table
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{name} Evaluation:")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC AUC': auc
    }

# Initialize models
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Evaluate all models
results = []
results.append(evaluate_model("Random Forest", clf, X_train, y_train, X_test, y_test))
results.append(evaluate_model("Logistic Regression", logreg, X_train, y_train, X_test, y_test))
results.append(evaluate_model("XGBoost", xgb, X_train, y_train, X_test, y_test))

# Show results comparison
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Table:")
print(results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Predict with all models
y_pred_rf = clf.predict(X_test)
y_pred_lr = logreg.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# Step 2: Plot confusion matrices
models = {
    "Random Forest": y_pred_rf,
    "Logistic Regression": y_pred_lr,
    "XGBoost": y_pred_xgb
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (model_name, y_pred) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title(model_name)

plt.suptitle("🔍 Confusion Matrices: Binary Classification Models", fontsize=16)
plt.tight_layout()
plt.show()

# Step 1: Select random attack + normal rows from test set
sample_attacks = X_test[y_test.values == 1].sample(2, random_state=1)
sample_normals = X_test[y_test.values == 0].sample(2, random_state=2)

# Combine into one small batch
sample_input = pd.concat([sample_attacks, sample_normals]).reset_index(drop=True)

# Step 2: Predict using Random Forest (you can switch to xgb or logreg if needed)
sample_pred = clf.predict(sample_input)
sample_proba = clf.predict_proba(sample_input)[:, 1]

# Step 3: Match with original details: risk + attack type (already stored in df_full)
# Reuse the same test indices
sample_indices = sample_input.index
full_test_df = df_full.iloc[train_len:].reset_index(drop=True)
sample_risks = full_test_df.loc[sample_indices, 'risk'].values
sample_attacks = full_test_df.loc[sample_indices, 'attack_cat'].values

# Step 4: Generate full response table
response_map = {
    'Low': 'Allow',
    'Medium': 'Alert + Monitor',
    'High': 'Alert + Block'
}

df_sample_results = pd.DataFrame({
    'Predicted_Label': sample_pred,
    'Attack_Category': sample_attacks,
    'Risk': sample_risks,
    'Confidence (Attack)': sample_proba,
})

df_sample_results['Suggested_Action'] = df_sample_results['Risk'].map(response_map)

# Step 5: Show results
print("\n🚦 Real-Time Sample Predictions:")
print(df_sample_results)

def predict_realtime_traffic(input_data, model, label_encoders, reference_columns, threshold=0.5):
    """
    Real-time prediction function with safe column handling.

    Parameters:
        input_data (dict): Dictionary of input features.
        model: Trained classifier.
        label_encoders (dict): Encoders for categorical features.
        reference_columns (list): Column names used during model training.
        threshold (float): Attack probability threshold.

    Returns:
        dict: Prediction label, confidence, and suggested action.
    """
    import pandas as pd

    # Convert to single-row DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical features safely
    for col in ['proto', 'service', 'state']:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Ensure all expected columns are present
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0  # default value for missing columns

    # Reorder columns to match training
    df = df[reference_columns]

    # Predict
    pred_prob = model.predict_proba(df)[0][1]
    pred_label = int(pred_prob >= threshold)

    # Map label to action
    if pred_label == 1 and pred_prob >= 0.9:
        action = "Alert + Block"
    elif pred_label == 1:
        action = "Alert + Monitor"
    else:
        action = "Allow"

    return {
        "Predicted_Label": "Attack" if pred_label == 1 else "Normal",
        "Confidence": round(pred_prob, 4),
        "Suggested_Action": action
    }

# Reference columns from training
reference_columns = X_train.columns.tolist()

# Your sample input
sample_input = {
    'dur': 0.5, 'proto': 'tcp', 'service': 'http', 'state': 'FIN',
    'spkts': 10, 'dpkts': 12, 'sbytes': 560, 'dbytes': 890,
    'rate': 120.0, 'sload': 2500.0, 'dload': 1500.0, 'sttl': 64, 'dttl': 63,
    'sloss': 0, 'dloss': 0, 'sinpkt': 0.01, 'dinpkt': 0.02,
    'sjit': 0.0, 'djit': 0.0, 'swin': 255, 'dwin': 255,
    'stcpb': 0, 'dtcpb': 0, 'tcprtt': 0.1, 'synack': 0.1, 'ackdat': 0.1,
    'smean': 60, 'dmean': 90, 'trans_depth': 0, 'response_body_len': 0,
    'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1,
    'is_ftp_login': 0, 'ct_ftp_cmd': 0, 'ct_flw_http_mthd': 0,
    'is_sm_ips_ports': 0
}

# Call updated real-time function
result = predict_realtime_traffic(sample_input, clf, label_encoders, reference_columns)
print("🚨 Real-Time Prediction Result:")
print(result)

pip install shap

import shap

# Preload SHAP explainer globally for speed (for tree models)
explainer = shap.TreeExplainer(clf)

def predict_realtime_with_shap(input_data, model, label_encoders, reference_columns, explainer, threshold=0.5):
    """
    Real-time prediction with SHAP-based feature explanation (top 3).

    Parameters:
        input_data (dict): Dictionary of feature values.
        model: Trained model.
        label_encoders: Dict of label encoders for categorical columns.
        reference_columns: List of columns used during training.
        explainer: Pre-initialized SHAP TreeExplainer object.
        threshold: Classification threshold for attack.

    Returns:
        dict: Label, confidence, suggested action, top 3 influential features
    """
    import pandas as pd

    # Step 1: Prepare input DataFrame
    df = pd.DataFrame([input_data])

    for col in ['proto', 'service', 'state']:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[reference_columns]

    # Step 2: Predict
    pred_prob = model.predict_proba(df)[0][1]
    pred_label = int(pred_prob >= threshold)

    # Step 3: Suggested action
    if pred_label == 1 and pred_prob >= 0.9:
        action = "Alert + Block"
    elif pred_label == 1:
        action = "Alert + Monitor"
    else:
        action = "Allow"

    # Step 4: SHAP values for explanation
    shap_values = explainer.shap_values(df)
    shap_contrib = shap_values[1] if isinstance(shap_values, list) else shap_values[0]

    feature_impact = list(zip(df.columns, shap_contrib[0]))
    top3 = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)[:3]
    top_features = [{ "feature": f, "impact": round(v, 4)} for f, v in top3]

    return {
        "Predicted_Label": "Attack" if pred_label == 1 else "Normal",
        "Confidence": round(pred_prob, 4),
        "Suggested_Action": action,
        "Top_3_Influential_Features": top_features
    }

result = predict_realtime_with_shap(sample_input, clf, label_encoders, reference_columns, explainer)
print("🚨 Real-Time Prediction + Explainability:")
print(result)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from trained model
importances = clf.feature_importances_
features = X_train.columns

# Create DataFrame
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(20)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

