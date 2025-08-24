import streamlit as st
from training import training_page
from prediction import prediction_page
from utils import create_models_dir

# Create models directory if it doesn't exist
create_models_dir()

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Training", "Prediction"])

if menu == "Training":
    training_page()
elif menu == "Prediction":
    prediction_page()