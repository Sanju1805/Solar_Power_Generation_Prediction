import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/solar_data.csv")

# Input features
X = data[
    [
        "Temperature",
        "Humidity",
        "Cloud_Cover",
        "Solar_Irradiance",
        "Wind_Speed"
    ]
]

# Target value
y = data["Power_Generation"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Training Completed Successfully!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/solar_model.pkl")

print("\nModel saved successfully!")
print("Location: models/solar_model.pkl")