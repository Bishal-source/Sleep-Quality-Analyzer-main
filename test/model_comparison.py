"""
Model Comparison: XGBoost, Random Forest, KNN, SVM, Decision Tree
Evaluates: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("=" * 70)
print("SLEEP QUALITY MODEL COMPARISON")
print("=" * 70)

# Load dataset
df = pd.read_csv(r"../dataset/sleep_realistic_final.csv")
print(f"\nOriginal dataset: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Data cleaning
df_clean = df[
    (df['Sleep Duration'] > 0) &
    (df['SpO2 Before'] >= 0) & (df['SpO2 Before'] <= 100) &
    (df['SpO2 After'] >= 0) & (df['SpO2 After'] <= 100) &
    (df['Heart Rate Before'] > 0) & (df['Heart Rate After'] > 0)
].copy()

# Feature engineering
df_clean['HR_Change'] = df_clean['Heart Rate After'] - df_clean['Heart Rate Before']
df_clean['SpO2_Change'] = df_clean['SpO2 After'] - df_clean['SpO2 Before']

print(f"Cleaned dataset: {len(df_clean)} rows")

# Encode target
quality_mapping = {'poor': 0, 'good': 1, 'best': 2}
df_clean['target'] = df_clean['Sleep Quality'].map(quality_mapping)

# Features - using all relevant features
features = ['Sleep Duration', 'Stress Level', 'Physical Activity',
            'Caffeine Cups', 'Screen Time', 'SpO2 Before', 'SpO2 After',
            'Heart Rate Before', 'Heart Rate After', 'Age', 'HR_Change', 'SpO2_Change']

X = df_clean[features]
y = df_clean['target']

# Class distribution
print("\nTarget distribution:")
print(df_clean['Sleep Quality'].value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for KNN and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Define models (excluding SVM)
models = {
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
}

# Store results
results = []

print("\n" + "=" * 70)
print("TRAINING AND EVALUATING MODELS")
print("=" * 70)

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    # Use scaled data for KNN and SVM
    if name in ['KNN', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0

    # Cross-validation
    if name in ['KNN', 'SVM']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV Mean': cv_mean,
        'CV Std': cv_std
    })

    # Print metrics
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc*100:.2f}%")
    print(f"\nCross-Validation: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['poor', 'good', 'best']))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")

# Summary table
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
    'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Mean', 'CV Std'))
print("-" * 87)

for _, row in results_df.iterrows():
    print("{:<15} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>11.2f}% {:>9.2f}%".format(
        row['Model'],
        row['Accuracy']*100,
        row['Precision']*100,
        row['Recall']*100,
        row['F1-Score']*100,
        row['ROC-AUC']*100,
        row['CV Mean']*100,
        row['CV Std']*100
    ))

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n\nResults saved to: model_comparison_results.csv")

# Identify best model
best_model = results_df.iloc[0]
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print(f"\nBest Model: {best_model['Model']}")
print(f"  Accuracy:  {best_model['Accuracy']*100:.2f}%")
print(f"  F1-Score:  {best_model['F1-Score']*100:.2f}%")
print(f"  ROC-AUC:   {best_model['ROC-AUC']*100:.2f}%")
print("=" * 70)