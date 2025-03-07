from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Load the trained model
MODEL_PATH = 'customer_churn_gbm_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            app.logger.error("Model could not be loaded.")
            raise HTTPException(description="The model could not be loaded. Please check the logs.", code=500)

        # Get form data from the frontend
        Account_length = float(request.form['Account_length'])
        International_plan = float(request.form['International_plan'])
        Number_vmail_messages = float(request.form['Number_vmail_messages'])
        Total_day_calls = float(request.form['Total_day_calls'])
        Total_day_charge = float(request.form['Total_day_charge'])
        Total_eve_calls = float(request.form['Total_eve_calls'])
        Total_eve_charge = float(request.form['Total_eve_charge'])
        Total_night_calls = float(request.form['Total_night_calls'])
        Total_night_charge = float(request.form['Total_night_charge'])
        Total_intl_calls = float(request.form['Total_intl_calls'])
        Total_intl_charge = float(request.form['Total_intl_charge'])
        Customer_service_calls = float(request.form['Customer_service_calls'])
        state = request.form['state']

        # Handle state encoding
        if state in ['AR', 'CA', 'MD', 'ME', 'MI', 'NJ', 'MS', 'MT', 'NV', 'SC', 'TX', 'WA']:
            state_value = 2
        elif state in ['AK', 'AL', 'AZ', 'DC', 'HI', 'IA', 'IL', 'LA', 'MO', 'MD', 'NE', 'NM', 'RI', 'TN', 'VA', 'VT', 'WI', 'WV', 'WY']:
            state_value = 0
        else:
            state_value = 1

        # Define the weights for the Usage Score
        weights = [0.4, 0.3, 0.2, 0.1]  # Example weights (adjust according to your actual values)

        # Calculate Usage Score
        usage_score = (
            Total_day_charge * weights[0] +
            Total_eve_charge * weights[1] +
            Total_night_charge * weights[2] +
            Total_intl_charge * weights[3]
        )

        features = np.array([Account_length, 
                             International_plan, 
                             Number_vmail_messages, 
                             Total_day_calls,
                             Total_day_charge,
                             Total_eve_calls,
                             Total_eve_charge,
                             Total_night_calls,
                             Total_night_charge,
                             Total_intl_calls,
                             Total_intl_charge,
                             Customer_service_calls,
                             state_value,
                             usage_score
                             ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        app.logger.info(f"Prediction made: {prediction[0]}, Probability: {probability}")

        # Redirect to a new page with the result
        return render_template('result.html', prediction=int(prediction[0]), churn_probability=probability)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return render_template('result.html', error=f"Error during prediction: {str(e)}")

# Handle HTTP exceptions
@app.errorhandler(HTTPException)
def handle_exception(error):
    # Log the error
    app.logger.error(f"HTTP Error {error.code}: {error.description}")
    # Return a JSON error response with the appropriate HTTP status code
    return jsonify({'error': error.description}), error.code

if __name__ == '__main__':
    app.run(port=5001)
