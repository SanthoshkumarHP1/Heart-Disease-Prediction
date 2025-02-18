from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Load trained model
model = pickle.load(open("model.pkl", "rb"))

# ✅ Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# ✅ Define feature order (must match training!)
FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
            'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
            'Education', 'Income']

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # ✅ Ensure prediction is stored and accessible after form submission

    if request.method == "POST":
        try:
            print("📝 Form data received:", request.form)  # Debugging output

            # ✅ Collect form data in correct order
            features = []
            for field in FEATURES:
                value = request.form.get(field, '').strip()  # Get user input
                print(f"Received {field}: {value}")  # Debugging input
                if value:
                    try:
                        features.append(float(value))  # Convert to float
                    except ValueError:
                        raise ValueError(f"Invalid input for {field}: {value} is not a number.")
                else:
                    raise ValueError(f"Missing input for: {field}")

            # ✅ Convert to numpy array
            features_array = np.array([features])

            # ✅ Apply scaling (must match training)
            features_df = pd.DataFrame(features_array, columns=FEATURES)  # Convert to DataFrame
            features_array_scaled = scaler.transform(features_df)

            # ✅ Make prediction
            prediction = model.predict(features_array_scaled)[0]
            print(f"✅ Prediction result: {prediction}")  # Debugging prediction result

            prediction = "🛑 Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease"

        except Exception as e:
            print(f"❌ Error: {e}")  # Debugging error message
            prediction = f"⚠️ Invalid Input: {e}"

    return render_template("index.html", prediction=prediction)  # ✅ Pass prediction to HTML

if __name__ == "__main__":
    app.run(debug=True)
