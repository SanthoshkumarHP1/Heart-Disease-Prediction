Heart Disease Prediction Model
This project implements a machine learning model to predict the likelihood of heart disease based on various health metrics. The heart disease prediction model is built using Logistic Regression, a commonly used algorithm for binary classification problems. The project also includes a Flask web application that provides an interactive interface for users to input their data and get predictions.

Features:
Heart Disease Prediction Model: A machine learning model trained on various health-related features like age, blood pressure, cholesterol levels, BMI, smoking habits, etc., to predict whether an individual is at risk of heart disease.
Flask Web Application: The model is integrated into a web application, allowing users to input relevant data via a simple HTML form and get predictions in real time.
User-Friendly Frontend: The project provides a clean and intuitive frontend to interact with the model. Users can easily enter their health details and see the prediction results instantly.
Model Evaluation: The model is trained and evaluated using various metrics, such as accuracy, precision, recall, and confusion matrix, to ensure the prediction reliability.
Technologies Used:
Machine Learning: Logistic Regression from the scikit-learn library for heart disease prediction.
Frontend: HTML, CSS, and JavaScript to create the user interface.
Backend: Flask for creating a lightweight web server to handle form submissions and serve predictions.
Data Processing: pandas and NumPy for preprocessing and data handling.
Model Saving and Loading: pickle for saving and loading the trained model (model.pkl).
Deployment: The project can be deployed on web platforms like Heroku, AWS, or DigitalOcean for public access.
