import os
import joblib

MODEL_PATH=os.path.join("data","model","best_model.pkl")
PREPROCESSOR_PATH=os.path.join("data","processed","preprocessor.pkl")
model=joblib.load(MODEL_PATH)
preprocessor=joblib.load(PREPROCESSOR_PATH)

def predict(input_data):
    processed_data=preprocessor.transform(input_data)
    prediction=model.predict(processed_data)[0]
    probability=model.predict_proba(processed_data)[0][1]
    return prediction , probability


