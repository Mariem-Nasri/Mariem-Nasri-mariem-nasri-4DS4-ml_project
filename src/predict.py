import joblib
import numpy as np
from src.config import DATA_PATHS

def make_prediction(gbm, logger):
    try:
        # Load the scaler
        scaler = joblib.load(DATA_PATHS["scaler"])
    except FileNotFoundError:
        logger.error("Scaler file not found. Ensure the scaler is saved and the correct path is used.")
        return

    # Prepare the input data (the same format as used in training)
    prediction_data = {
        'Account_length': 100, 'International_plan': 1, 'Number_vmail_messages': 25,
        'Total_day_calls': 150, 'Total_day_charge': 45.5, 'Total_eve_calls': 130,
        'Total_eve_charge': 35.7, 'Total_night_calls': 120, 'Total_night_charge': 30.2,
        'Total_intl_calls': 30, 'Total_intl_charge': 10.5, 'Customer_service_calls': 2,
        'State': 'CA'
    }

    # Mapping the 'State' to numerical (Ensure unique values for each state)
    state_mapping = {"CA": 2, "TX": 3, "NY": 1, "FL": 0}
    state_value = state_mapping.get(prediction_data['State'], 1)  # Default to 1 if not in the mapping

    # Usage score calculation (same as during training)
    weights = [0.4, 0.3, 0.2, 0.1]
    usage_score = (
        prediction_data['Total_day_charge'] * weights[0] +
        prediction_data['Total_eve_charge'] * weights[1] +
        prediction_data['Total_night_charge'] * weights[2] +
        prediction_data['Total_intl_charge'] * weights[3]
    )

    # Creating feature vector
    features = np.array([
        prediction_data['Account_length'], prediction_data['International_plan'],
        prediction_data['Number_vmail_messages'], prediction_data['Total_day_calls'],
        prediction_data['Total_day_charge'], prediction_data['Total_eve_calls'],
        prediction_data['Total_eve_charge'], prediction_data['Total_night_calls'],
        prediction_data['Total_night_charge'], prediction_data['Total_intl_calls'],
        prediction_data['Total_intl_charge'], prediction_data['Customer_service_calls'],
        state_value, usage_score
    ]).reshape(1, -1)

    # Scaling the input data
    features_scaled = scaler.transform(features)

    # Making prediction
    prediction = gbm.predict(features_scaled)
    probability = gbm.predict_proba(features_scaled)[0][1]  # Probability for churn (class 1)

    # Log results
    logger.info(f"Prediction: {prediction[0]}, Probability: {probability}")
    print(f"Prediction: {prediction[0]}")
    print(f"Churn Probability: {probability:.4f}")

