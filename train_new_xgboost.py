import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING NEW XGBOOST MODEL")
print("=" * 60)

# Load dataset
df = pd.read_csv(r"dataset/sleep_quality_dataset.csv")
print(f"Original data: {len(df)} rows")

# Clean data
df_clean = df[
    (df['Sleep Duration'] > 0) &
    (df['SpO2 Before'] >= 0) & (df['SpO2 Before'] <= 100) &
    (df['SpO2 After'] >= 0) & (df['SpO2 After'] <= 100) &
    (df['Heart Rate Before'] > 0) & (df['Heart Rate After'] > 0)
].copy()

# Encode target: poor=0, good=1, best=2
quality_mapping = {'poor': 0, 'good': 1, 'best': 2}
df_clean['target'] = df_clean['Sleep Quality'].map(quality_mapping)

# Feature engineering: HR_Change
df_clean['HR_Change'] = df_clean['Heart Rate After'] - df_clean['Heart Rate Before']

print(f"Clean data: {len(df_clean)} rows")

# Features for new model
features = ['Sleep Duration', 'Stress Level', 'Physical Activity',
            'Caffeine Cups', 'Screen Time', 'HR_Change']

X = df_clean[features]
y = df_clean['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# Train XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['poor', 'good', 'best']))

# Save model
model.save_model("xgboost_model.json")
print("\nModel saved to: xgboost_model.json")

# Save feature names
feature_info = {
    'features': features,
    'mapping': quality_mapping
}
with open("feature_info.pkl", "wb") as f:
    pickle.dump(feature_info, f)

print("Feature info saved to: feature_info.pkl")
print("\nDone!")