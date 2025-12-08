# ===============================================
# HEART DISEASE ANALYSIS AND PREDICTION
# ===============================================

# 1. Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ---------------------------
# a. Load the data
# ---------------------------
df = pd.read_csv(r"C:\Users\Neeka Javeed\Desktop\tools project\heart_disease_uci.csv")
print("First 5 rows of dataset:\n", df.head())
print("\nDataset info:\n")
print(df.info())
print("\nDataset statistics:\n", df.describe())
print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns.")
print("Target variable is 'num', indicating presence of heart disease.\n")

# ---------------------------
# b. Exploratory Data Analysis (EDA)
# ---------------------------

print("Missing values per column:\n", df.isnull().sum())
print("\nSummary statistics for numeric features:\n")
print(df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x='num', data=df, palette='Set2')
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease Presence (num)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, bins=20, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df, palette='pastel')
plt.title("Sex Distribution")
plt.xticks([0,1], ['Female','Male'])
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='num', data=df, palette='Set1')
plt.title("Heart Disease by Sex")
plt.xticks([0,1], ['Female','Male'])
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='thalch', hue='num', data=df, palette='coolwarm')
plt.title("Age vs Max Heart Rate by Heart Disease")
plt.show()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12,10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# ---------------------------
# c. Data Wrangling / Cleansing
# ---------------------------

df.fillna(df.median(numeric_only=True), inplace=True)
df['fbs'] = df['fbs'].fillna(0).astype(int)
df['exang'] = df['exang'].fillna(0).astype(int)

df['sex'] = df['sex'].map({'Male':1, 'Female':0})

categorical_cols = ['cp','restecg','slope','thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

num_features = ['age','trestbps','chol','thalch','oldpeak']
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

df.drop_duplicates(inplace=True)
print(f"Dataset shape after preprocessing: {df.shape}")

df.drop('dataset', axis=1, inplace=True)

# ---------------------------
# d. Build multiple charts
# ---------------------------

plt.figure(figsize=(6,4))
sns.boxplot(x='num', y='age', data=df, palette='Set3')
plt.title("Age vs Heart Disease")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df[df['num']==1]['chol'], color='red', label='Disease', kde=True)
sns.histplot(df[df['num']==0]['chol'], color='green', label='No Disease', kde=True)
plt.title("Cholesterol vs Heart Disease")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='num', y='thalch', data=df, palette='Set2')
plt.title("Max Heart Rate vs Heart Disease")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='num', y='oldpeak', data=df, palette='Set1')
plt.title("ST Depression (oldpeak) vs Heart Disease")
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title("Correlation Heatmap (All Numeric Features Encoded)")
plt.show()

selected_features = ['age','thalch','chol','oldpeak','trestbps','num']
sns.pairplot(df[selected_features], hue='num', palette='coolwarm')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

# ---------------------------
# f. Machine Learning Models
# ---------------------------

X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Store model scores
model_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name} Metrics:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 3))
    print("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 3))
    print("F1-score:", round(f1_score(y_test, y_pred, average='weighted'), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save accuracy for comparison plot
    model_scores[name] = accuracy_score(y_test, y_pred)

    # -------- Confusion Matrix Heatmap --------
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # -------- Classification Report --------
    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

# ---------------------------
# g. Model Comparison Chart
# ---------------------------

plt.figure(figsize=(7,5))
sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette='coolwarm')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0,1)
plt.show()

# ---------------------------
# h. ROC Curve for All Models
# ---------------------------

plt.figure(figsize=(7,6))

for name, model in models.items():
    try:
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    except:
        pass  # Skip models without predict_proba()

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
