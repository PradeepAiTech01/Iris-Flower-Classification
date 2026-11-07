from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

df = pd.read_csv(url)

# Display first 5 rows
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display flower categories
print("\nFlower Species:")
print(df["species"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Statistical analysis
print("\nStatistical Summary:")
print(df.describe())

# 2. Species distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="species")
plt.title("Distribution of Iris Species")
plt.xlabel("Flower Species")
plt.ylabel("Count")
plt.show()

# 3. Pairplot
sns.pairplot(df, hue="species")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(8, 6))
correlation = df.drop(columns=["species"]).corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Separate features and target
X = df.drop("species", axis=1)
y = df["species"]

# Convert species into numerical values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Encoded Species:")
print(encoder.classes_)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Display dataset information
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

print("\nData Preprocessing Completed Successfully!")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

# Train and evaluate models
results = {}

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name}: {accuracy * 100:.2f}%")

print("\nModel Training Completed!")
# Model Evaluation

for name, model in models.items():

    # Predict test data
    y_pred = model.predict(X_test)

    print("\n================================")
    print("Model:", name)
    print("================================")

    # Calculate evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    display.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    # Final Model Comparison

comparison = pd.DataFrame({
    "Model": list(models.keys()),
    "Accuracy": [
        accuracy_score(y_test, models[name].predict(X_test))
        for name in models
    ]
})

comparison["Accuracy"] = comparison["Accuracy"] * 100

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# Save results
comparison.to_csv("model_comparison.csv", index=False)

print("\nComparison saved successfully!")
import joblib

# Save the best model
joblib.dump(models["SVM"], "iris_svm_model.pkl")

# Save scaler
joblib.dump(scaler, "iris_scaler.pkl")

# Save label encoder
joblib.dump(encoder, "iris_encoder.pkl")

print("\nBest SVM Model Saved Successfully!")
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare accuracy comparison
comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("model_accuracy_comparison.png")
plt.show()

print("Model comparison visualization saved successfully!")
# STEP 7: MODEL ACCURACY VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns

# Sort models by accuracy
comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

# Create bar chart
plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=20)

plt.tight_layout()

# Save graph
plt.savefig("model_accuracy_comparison.png")

plt.show()

print("Model comparison visualization saved successfully!")