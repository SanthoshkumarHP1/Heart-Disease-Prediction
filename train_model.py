import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ✅ Load dataset (Ensure it contains only the 21 features)
heart_disease = pd.read_csv("heart_disease.csv")

# ✅ Select the correct features (21 features)
FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
            'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
            'Education', 'Income']

X = heart_disease[FEATURES]  # Only 21 features
y = heart_disease['HeartDiseaseorAttack']  # Target variable

# ✅ Apply scaling to normalize data (if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# ✅ Save the scaler and model
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and scaler saved successfully!")
