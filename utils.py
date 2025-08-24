import os
import joblib
from datetime import datetime

def create_models_dir():
    if not os.path.exists('models'):
        os.makedirs('models')

def get_model_files():
    return [f for f in os.listdir('models') if f.endswith('.pkl')]

def load_model(selected_model):
    model_path = os.path.join('models', selected_model)
    return joblib.load(model_path)

def save_model(pipeline, file_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/{file_name}_{timestamp}.pkl"
    joblib.dump(pipeline, model_filename)
    return model_filename  # Return the filename for potential display