import joblib
import os

model = joblib.load(os.path.join(os.path.dirname(__file__), "models/sentiment_pipe.joblib"))
print(model)