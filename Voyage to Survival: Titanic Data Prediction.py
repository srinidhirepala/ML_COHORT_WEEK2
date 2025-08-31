# Titanic EDA + Machine Learning Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===========================
# Load Dataset
# ===========================
titanic = sns.load_dataset("titanic")

# Check missing values
print("Missing Values Before Cleaning:\n", titanic.isnull().sum())

# Drop useless columns
titanic = titanic.drop(['deck','embark_town','alive','class','who','adult_male'], axis=1)

# Fill missing values
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Encode categorical variables
label_enc = LabelEncoder()
for col in titanic.select_dtypes(include='object').columns:
    titanic[col] = label_enc.fit_transform(titanic[col])

print("\nData After Cleaning & Encoding:\n")
titanic.info()

# ===========================
# Exploratory Data Analysis
# ===========================

# Survival Count
sns.countplot(data=titanic, x="survived")
plt.title("Survival Count (0=Not Survived, 1=Survived)")
plt.show()

# Survival by Gender
sns.countplot(data=titanic, x="sex", hue="survived")
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
sns.countplot(data=titanic, x="pclass", hue="survived")
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
sns.histplot(titanic['age'], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(titanic.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ===========================
# Machine Learning Models
# ===========================

# Features & Target
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# ===========================
# Model Accuracy Comparison
# ===========================
models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt,
    "Random Forest": rf
}

results = {}
for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.show()
