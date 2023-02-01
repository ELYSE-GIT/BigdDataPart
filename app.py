import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model and scaler
lgb_credits = joblib.load('src/model/lgb_credits.joblib')
scaler = joblib.load('src/model/minMax_scaler_credits.joblib')

# Define the features used in the model
used_features = [
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'CNT_FAM_MEMBERS',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

# Define Flask App
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    transformed_data = scaler.transform(data)

    # Make prediction using model loaded from disk as per the data.
    prediction = lgb_credits.predict(transformed_data)

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    app.run(port=8000, debug=True)

