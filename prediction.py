import streamlit as st
import pandas as pd
import os
from utils import get_model_files, load_model

def prediction_page():
    st.title("Model Prediction")

    # Select model from models folder
    model_files = get_model_files()
    if not model_files:
        st.warning("No trained models found in the 'models' folder.")
    else:
        selected_model = st.selectbox("Select Trained Model", model_files)

        # Data upload for prediction
        pred_file = st.file_uploader("Upload CSV data for prediction", type="csv", key="pred_file")

        if pred_file is not None:
            pred_df = pd.read_csv(pred_file)
            st.write("Prediction Data Preview:")
            st.dataframe(pred_df.head())

            # Prediction button
            if st.button("Start Prediction"):
                try:
                    # Load selected model
                    model = load_model(selected_model)

                    # Make predictions
                    predictions = model.predict(pred_df)

                    # Display predictions as a table
                    pred_df['Prediction'] = predictions
                    st.write("Prediction Results:")
                    st.dataframe(pred_df)

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")