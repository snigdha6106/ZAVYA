import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# -------------------------
# 1. Load Dataset
# -------------------------
DATA_FILE = 'student_quiz_dataset_v2.csv'
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please run create_dataset.py first.")
    data = {
        'Python': [50], 'OOP': [50], 'Data Structures': [50], 'DSA': [50], 
        'Java': [50], 'AI/ML': [50], 'OS': [50], 'Skill_Level': ['beginner']
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(DATA_FILE)

subjects = ['Python','OOP','Data Structures','DSA','Java','AI/ML','OS']

# Ensure all subject columns exist, fill with 0 if not
for subj in subjects:
    if subj not in df.columns:
        df[subj] = 0

# --- Feature Engineering ---
df['total_score'] = df[subjects].sum(axis=1)
df['std_dev_scores'] = df[subjects].std(axis=1).fillna(0)
df['num_weak'] = (df[subjects] < 50).sum(axis=1)
df['num_strong'] = (df[subjects] > 80).sum(axis=1)
df['range'] = df[subjects].max(axis=1) - df[subjects].min(axis=1)

features = subjects + ['total_score', 'std_dev_scores', 'num_weak', 'num_strong', 'range']
X = df[features]
# --- End Feature Engineering ---

le = LabelEncoder()
if 'Skill_Level' not in df.columns:
    df['Skill_Level'] = 'beginner'
y = le.fit_transform(df['Skill_Level'])

# -------------------------
# 2. Train-Test Split
# -------------------------
if len(df) < 5:
    X_train, X_test, y_train, y_test = X, X, y, y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# 3. Handle Class Imbalance using SMOTE
# -------------------------
try:
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
except ValueError:
    print("Warning: Could not apply SMOTE. Training on original data.")
    X_train_res, y_train_res = X_train, y_train

# -------------------------
# 4. Train FINAL Model
# -------------------------
print("Training the final model...")
model = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_res, y_train_res)
print("Model trained successfully.")

# -------------------------
# 5. Evaluate Model (for console)
# -------------------------
if __name__ == "__main__":
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("\n===== FINAL MODEL METRICS =====")
        print(f"Accuracy: {acc*100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (Acc: {acc*100:.2f}%)')
        plt.show() 

        # Feature Importance
        feat_importance_data = {
            'Feature': features, 
            'Importance': model.feature_importances_
        }
        feat_importance = pd.DataFrame(feat_importance_data).sort_values(by='Importance', ascending=False)

        print("\nFeature Importance:")
        print(feat_importance)
    else:
        print("Not enough data to create a test split.")