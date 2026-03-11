import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# LOAD NEW DATASET WITH GOOD, AVERAGE, BAD LABELS

df = pd.read_csv(
    "dataset/sleep_quality_dataset.csv"
)


print("Dataset loaded")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Label distribution:")
print(df["Sleep Quality"].value_counts())


# FEATURES (must match your web app inputs)

X = df[[
    "Sleep Duration",
    "Stress Level",
    "Physical Activity",
    "Caffeine Intake",
    "Screen Time",
    "SpO2 Before",
    "SpO2 After",
    "Heart Rate Before",
    "Heart Rate After"
]]


# TARGET

y = df["Sleep Quality"]


# ENCODE TARGET

le = LabelEncoder()

y = le.fit_transform(y)

print("\nLabel classes:", le.classes_)


# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# TRAIN MODEL

model = RandomForestClassifier(

    n_estimators=500,
    max_depth=12,
    random_state=42

)

model.fit(X_train, y_train)


# SAVE MODEL + ENCODER

pickle.dump(
    model,
    open("sleep_model.pkl", "wb")
)

pickle.dump(
    le,
    open("label_encoder.pkl", "wb")
)


# ACCURACY

accuracy = model.score(X_test, y_test)

print("\nModel trained successfully")
print("Accuracy:", accuracy)