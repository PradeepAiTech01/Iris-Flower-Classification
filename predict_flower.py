import joblib
import numpy as np

# Load saved files
model = joblib.load("iris_svm_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
encoder = joblib.load("iris_encoder.pkl")

print("IRIS FLOWER PREDICTION SYSTEM")

# Get user input
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

# Prepare input
import pandas as pd

data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]], columns=[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
])

# Scale input
data_scaled = scaler.transform(data)

# Predict
prediction = model.predict(data_scaled)

# Convert numerical output to species name
flower = encoder.inverse_transform(prediction)

print("\nPredicted Flower Species:", flower[0])