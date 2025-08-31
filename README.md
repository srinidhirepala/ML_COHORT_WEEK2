# ML_COHORT_WEEK2

# 🛳️ Titanic Dataset - Exploratory Data Analysis & Machine Learning

This project performs **Exploratory Data Analysis (EDA)** and applies different **Machine Learning models** on the Titanic dataset to predict passenger survival.

---

## 📌 Dataset
We use the **Titanic dataset** available in `seaborn`:
```python
import seaborn as sns
titanic = sns.load_dataset("titanic")


##📊 Exploratory Data Analysis (EDA)

Handle missing values

Drop irrelevant columns

Encode categorical variables

Visualize:

Survival counts

Survival by gender

Survival by passenger class

Age distribution

Correlation heatmap

##🤖 Machine Learning Models

We train and compare three models:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Metrics Used:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Model Comparison:

A bar chart displays model accuracies.

