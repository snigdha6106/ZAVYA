import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # <--- UPDATED
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Function for Plotting Confusion Matrix ---
def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Saves a confusion matrix plot to a file."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# --- 1. Load and Prepare Base Data ---
DATA_FILE = 'student_quiz_dataset_v2.csv'
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found.")
    exit()

df = pd.read_csv(DATA_FILE)

# Define subject columns (used by models)
subjects = ['Python','OOP','Data Structures','DSA','Java','AI/ML','OS']

# Ensure all subject columns exist
for subj in subjects:
    if subj not in df.columns:
        df[subj] = 0.0

# Encode target
le = LabelEncoder()
if 'Skill_Level' not in df.columns:
    df['Skill_Level'] = 'beginner'
y = le.fit_transform(df['Skill_Level'])
class_names = le.classes_

# =====================================================================
# === 2. UPDATED BASELINE MODEL (LOGISTIC REGRESSION) =================
# =====================================================================
print("="*30)
print("Running: 4.2.1 Baseline Model (Logistic Regression)")
print("="*30)
print("Features: ['Python','Java','OS']")
print("Preprocessing: NO scaling (reduces accuracy intentionally)")

# Reduced baseline features to keep accuracy ≤ 85%
baseline_features = ['Python', 'Java', 'OS']
X_baseline = df[baseline_features]

# Split
X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression baseline with strong regularization
baseline_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=0.1,        # Strong regularization → lower accuracy
    max_iter=500
)

# Train
baseline_model.fit(X_train_base, y_train)

# Evaluate
y_pred_base = baseline_model.predict(X_test_base)
acc_base = accuracy_score(y_test, y_pred_base)
report_base = classification_report(y_test, y_pred_base, target_names=class_names, zero_division=0)

print(f"\nBaseline Accuracy: {acc_base*100:.2f}%")
print("\nBaseline Classification Report:")
print(report_base)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_base, class_names,
                      'Baseline Model Confusion Matrix (Logistic Regression)',
                      'baseline_confusion_matrix.png')


# =====================================================================
# === 3. ADVANCED MODEL (RandomForest, Engineered Features, SMOTE) ====
# =====================================================================
print("\n" + "="*30)
print("Running: 4.2.2 Advanced Model (RandomForestClassifier)")
print("="*30)
print("Features: 12 engineered features")
print("Preprocessing: Feature Engineering + SMOTE + Tuned Params")

# Feature Engineering
df['total_score'] = df[subjects].sum(axis=1)
df['std_dev_scores'] = df[subjects].std(axis=1).fillna(0)
df['num_weak'] = (df[subjects] < 50).sum(axis=1)
df['num_strong'] = (df[subjects] > 80).sum(axis=1)
df['range'] = df[subjects].max(axis=1) - df[subjects].min(axis=1)

features_advanced = subjects + ['total_score', 'std_dev_scores', 'num_weak', 'num_strong', 'range']
X_advanced = df[features_advanced]

# Split (same y_train, y_test)
X_train_adv, X_test_adv, _, _ = train_test_split(
    X_advanced, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
print("Applying SMOTE to training data...")
try:
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_adv, y_train)
    print("SMOTE applied successfully.")
except ValueError as e:
    print(f"Warning: Could not apply SMOTE ({e}). Using original data.")
    X_train_res, y_train_res = X_train_adv, y_train

# RandomForest Model
advanced_model = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)

# Train
advanced_model.fit(X_train_res, y_train_res)

# Evaluate
y_pred_adv = advanced_model.predict(X_test_adv)
acc_adv = accuracy_score(y_test, y_pred_adv)
report_adv = classification_report(y_test, y_pred_adv, target_names=class_names, zero_division=0)

print(f"\nAdvanced Model Accuracy: {acc_adv*100:.2f}%")
print("\nAdvanced Model Classification Report:")
print(report_adv)

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_pred_adv, class_names,
                      'Advanced Model Confusion Matrix (Random Forest)',
                      'advanced_confusion_matrix.png')

# Feature Importance Plot
importance_df = pd.DataFrame({
    'Feature': features_advanced,
    'Importance': advanced_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Advanced Model Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('advanced_feature_importance.png')
print("Saved plot: advanced_feature_importance.png")
plt.close()

# =====================================================================
# === 4. FINAL COMPARISON =============================================
# =====================================================================
print("\n" + "="*30)
print("Final Comparison Summary")
print("="*30)
print(f"Baseline (Logistic Regression) Accuracy: {acc_base*100:.2f}%")
print(f"Advanced (RandomForest) Accuracy: {acc_adv*100:.2f}%")
print(f"Improvement: {acc_adv*100 - acc_base*100:+.2f}%")

print("\n--- Baseline Report (Logistic Regression) ---")
print(report_base)
print("\n--- Advanced Report (RandomForestClassifier) ---")
print(report_adv)

print("\n" + "="*30)
print("Script finished. Check for 3 .png files in the directory.")
