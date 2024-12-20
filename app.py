from flask import Flask, request, render_template, flash
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flashing messages

# Load the trained model
model = joblib.load('diabetes_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from the form
            gender = request.form['gender']
            age = request.form['age']
            hypertension = request.form['hypertension']
            heart_disease = request.form['heart_disease']
            smoking_history = request.form['smoking_history']
            bmi = request.form['bmi']
            HbA1c_level = request.form['HbA1c_level']
            blood_glucose_level = request.form['blood_glucose_level']

            # Validate input
            if not all([gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]):
                flash('Please fill in all fields.')
                return render_template('index.html')

            # Convert to appropriate types
            age = float(age)
            hypertension = int(hypertension)
            heart_disease = int(heart_disease)
            bmi = float(bmi)
            HbA1c_level = float(HbA1c_level)
            blood_glucose_level = float(blood_glucose_level)

            # Prepare the data for prediction as a DataFrame
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'smoking_history': [smoking_history],
                'bmi': [bmi],
                'HbA1c_level': [HbA1c_level],
                'blood_glucose_level': [blood_glucose_level]
            })

            # Make the prediction
            prediction = model.predict(input_data)

            # Render the result
            result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
            return render_template('result.html', prediction=result)
        
        except ValueError as e:
            flash(f"Invalid input: {str(e)}")
            return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
