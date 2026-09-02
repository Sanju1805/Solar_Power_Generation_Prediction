from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model_path = "models/solar_model.pkl"

model = None

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Solar Power Prediction Model Loaded Successfully!")
else:
    print("Model not found. Please run train_model.py first.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        if model is None:
            return jsonify({
                "error": "Model not found. Please train the model first."
            })

        data = request.get_json()

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        cloud_cover = float(data["cloud_cover"])
        solar_irradiance = float(data["solar_irradiance"])
        wind_speed = float(data["wind_speed"])

        features = np.array([[
            temperature,
            humidity,
            cloud_cover,
            solar_irradiance,
            wind_speed
        ]])

        prediction = model.predict(features)

        return jsonify({
            "prediction": round(float(prediction[0]), 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)