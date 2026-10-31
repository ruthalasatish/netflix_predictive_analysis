#IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# LOADING THE DATASET

df = pd.read_csv("C:/Users/V/Desktop/spy/netflix_customer_churn.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# DATA CLEANING

print("\nMissing Values Before Cleaning:")
print(df.isna().sum())

print("\nDuplicate Records Before Cleaning:")
print(df.duplicated().sum())

df = df.drop_duplicates()

# Removing Rows with Missing Customer ID or Churn Label
df = df.dropna(subset=["customer_id", "churned"])

# Handling Missing Numerical Values
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Handling Missing Categorical Values
for col in df.select_dtypes(include=["str", "object"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isna().sum())

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nChurn Distribution:")
print(df["churned"].value_counts())


#EXPLORATORY DATA ANALYSIS (EDA)

print("\nNumber of Rows and Columns:")
print(df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nLast Five Records:")
print(df.tail())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nChurn Counts:")
print(df["churned"].value_counts())

print("\nChurn Percentages:")
print(df["churned"].value_counts(normalize=True) * 100)

print("\nUnique Values in Each Column:")
print(df.nunique())

#DATA VISUALIZATION

# GRAPH 1: CHURN DISTRIBUTION
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="churned", hue="churned", palette=["teal", "coral"], legend=False)
plt.title("Netflix Customer Churn Distribution")
plt.xlabel("Churned (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# GRAPH 2: CORRELATION HEATMAP
numeric_df = df.select_dtypes(include=np.number).drop(columns=["customer_id"], errors="ignore")
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="magma", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# GRAPH 3: CUSTOMER AGE DISTRIBUTION

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=20, kde=True, color="purple")
plt.title("Netflix Customer Age Distribution")
plt.xlabel("Customer Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

#ENCODING CATEGORICAL DATA

df_encoded = pd.get_dummies(df, columns=["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"], dtype=int)
print("\nEncoded Dataset:")
print(df_encoded.head())

print("\nEncoded Dataset Information:")
df_encoded.info()

print("\nDataset Shape After Encoding:")
print(df_encoded.shape)


# SPLITTING AND SCALING THE DATA

X = df_encoded.drop(columns=["customer_id", "churned"])
y = df_encoded["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


#LOGISTIC REGRESSION

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, zero_division=0)
recall_lr = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)

print("\nLogistic Regression Results")
print("Accuracy:", round(accuracy_lr * 100, 2), "%")
print("Precision:", round(precision_lr, 3))
print("Recall:", round(recall_lr, 3))
print("F1-Score:", round(f1_lr, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))


# K-NEAREST NEIGHBORS (KNN)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, zero_division=0)
recall_knn = recall_score(y_test, y_pred_knn, zero_division=0)
f1_knn = f1_score(y_test, y_pred_knn, zero_division=0)

print("\nK-Nearest Neighbors Results")
print("Accuracy:", round(accuracy_knn * 100, 2), "%")
print("Precision:", round(precision_knn, 3))
print("Recall:", round(recall_knn, 3))
print("F1-Score:", round(f1_knn, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))


# DECISION TREE

dt = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, zero_division=0)
recall_dt = recall_score(y_test, y_pred_dt, zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)

print("\nDecision Tree Results")
print("Accuracy:", round(accuracy_dt * 100, 2), "%")
print("Precision:", round(precision_dt, 3))
print("Recall:", round(recall_dt, 3))
print("F1-Score:", round(f1_dt, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))


# RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print("\nRandom Forest Results")
print("Accuracy:", round(accuracy_rf * 100, 2), "%")
print("Precision:", round(precision_rf, 3))
print("Recall:", round(recall_rf, 3))
print("F1-Score:", round(f1_rf, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# SUPPORT VECTOR MACHINE (SVM)

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm, zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, zero_division=0)

print("\nSupport Vector Machine Results")
print("Accuracy:", round(accuracy_svm * 100, 2), "%")
print("Precision:", round(precision_svm, 3))
print("Recall:", round(recall_svm, 3))
print("F1-Score:", round(f1_svm, 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))



# COMPARISON OF ALL FIVE MODELS

results = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"],
    "Accuracy": [accuracy_lr, accuracy_knn, accuracy_dt, accuracy_rf, accuracy_svm],
    "Precision": [precision_lr, precision_knn, precision_dt, precision_rf, precision_svm],
    "Recall": [recall_lr, recall_knn, recall_dt, recall_rf, recall_svm],
    "F1-Score": [f1_lr, f1_knn, f1_dt, f1_rf, f1_svm]
})

print("\nComparison of All Models:")
print(results.round(3))

print("\nModel with Highest Accuracy:")
print(results.loc[results["Accuracy"].idxmax(), ["Model", "Accuracy"]])


#ACCURACY COMPARISON OF ALL FIVE MODELS

plt.figure(figsize=(10, 6))
sns.barplot(data=results, x="Model", y="Accuracy", hue="Model", palette="viridis", legend=False)
plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# K-MEANS CLUSTERING

scaler_cluster = MinMaxScaler()
X_cluster = scaler_cluster.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)
df["Cluster"] = clusters
print("\nK-Means Clustering Results:")
print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())
print("\nChurn Distribution Across Clusters:")
print(pd.crosstab(df["Cluster"], df["churned"]))

# FINAL SUMMARY

print("NETFLIX CUSTOMER CHURN PREDICTION: ")

print("\nTotal Customers Analyzed:", len(df))

print("\nTotal Customers Who Churned:")
print((df["churned"] == 1).sum())

print("\nTotal Customers Who Did Not Churn:")
print((df["churned"] == 0).sum())

print("\nChurn Percentage:")
print(round(df["churned"].mean() * 100, 2), "%")

print("\nFinal Model Comparison:")
print(results.round(3).to_string(index=False))

best_model = results.loc[results["Accuracy"].idxmax()]

print("\nBest Performing Model:", best_model["Model"])
print("Best Model Accuracy:", round(best_model["Accuracy"] * 100, 2), "%")
