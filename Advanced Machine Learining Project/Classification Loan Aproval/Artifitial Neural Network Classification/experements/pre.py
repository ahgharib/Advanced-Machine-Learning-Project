import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import load
scaler_path = "C:/Users/PC/OneDrive/Desktop/Try_ann2/scaler.pkl"
model_path = "C:/Users/PC/OneDrive/Desktop/Try_ann2/ANN.joblib"

def preprocess_input(input_df):
    input_df['education'] = input_df['education'].map({'Graduate': 0, 'Not Graduate': 1})
    input_df['self_employed'] = input_df['self_employed'].map({'Yes': 0, 'No': 1})
    input_df = input_df.drop(columns=['loan_id'])
    return input_df

def predict_status(scaler_path, model_path, prediction_input):
    scaler = joblib.load(scaler_path)

    model = load(model_path)

    scaled_input = scaler.transform(prediction_input)

    prediction = model.predict(scaled_input)

    pred = np.argmax(prediction)

    label_encoder_y = LabelEncoder()

    label_encoder_y.fit(['REJECTED', 'APPROVED'])

    predicted_status = label_encoder_y.inverse_transform([pred])

    return predicted_status










df = 1

input_df = pd.DataFrame(df)
preprocessed_input = preprocess_input(input_df)
input_df1 = pd.DataFrame(df)
preprocessed_input1 = preprocess_input(input_df1)
input_df2 = pd.DataFrame(df)
preprocessed_input2 = preprocess_input(input_df2)


predicted_status = predict_status(scaler_path, model_path, preprocessed_input)
predicted_status = predict_status(scaler_path, model_path, preprocessed_input1)
predicted_status = predict_status(scaler_path, model_path, preprocessed_input2)
