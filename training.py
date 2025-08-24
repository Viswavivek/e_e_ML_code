import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from utils import save_model

def training_page():
    st.title("Model Training")

    # Problem type
    problem_type = st.selectbox("Problem Type", ["Regression", "Classification"])

    # Algorithm selection based on problem type
    if problem_type == "Regression":
        algo_options = ["Linear", "RandomForest"]
    else:
        algo_options = ["Logistic", "RFClassifier"]

    algorithm = st.selectbox("Algorithm", algo_options)

    # Data upload
    uploaded_file = st.file_uploader("Upload your CSV data", type="csv", key="train_file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Display columns and their data types
        st.write("Column Data Types:")
        col_dtypes = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
        col_dtypes.columns = ['Column', 'Data Type']
        st.dataframe(col_dtypes)

        # Target column selection
        target_col = st.selectbox("Target Column", df.columns)

        # Feature selection for training
        feature_cols = st.multiselect("Select Features for Training", df.columns.drop(target_col))
        if not feature_cols:
            st.warning("Please select at least one feature for training.")
        else:
            # Allow changing data types for selected features only
            st.write("Change Column Data Types (if needed):")
            dtype_changes = {}
            for col in feature_cols:
                current_dtype = str(df[col].dtype)
                new_dtype = st.selectbox(
                    f"Data type for {col} (current: {current_dtype})",
                    options=["No Change", "int64", "float64", "object", "category"],
                    key=f"dtype_{col}"
                )
                if new_dtype != "No Change":
                    dtype_changes[col] = new_dtype

            # Apply data type changes
            for col, dtype in dtype_changes.items():
                try:
                    if dtype == "int64":
                        df[col] = pd.to_numeric(df[col], downcast="integer")
                    elif dtype == "float64":
                        df[col] = pd.to_numeric(df[col], downcast="float")
                    elif dtype == "object":
                        df[col] = df[col].astype(str)
                    elif dtype == "category":
                        df[col] = df[col].astype("category")
                    st.write(f"Changed {col} to {dtype}")
                except Exception as e:
                    st.error(f"Error changing {col} to {dtype}: {str(e)}")

            # Update data types display after changes
            if dtype_changes:
                st.write("Updated Column Data Types:")
                st.dataframe(pd.DataFrame(df[feature_cols].dtypes, columns=['Data Type']).reset_index())

            # Columns for One Hot Encoding
            ohe_cols = st.multiselect("Columns for One Hot Encoding", feature_cols)

            # Columns for Scaling
            scale_cols = st.multiselect("Columns for Scaling", [col for col in feature_cols if col not in ohe_cols])

            # Training button
            if st.button("Train Model"):
                # Prepare features and target
                X = df[feature_cols]
                y = df[target_col]

                # Preprocessing pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols),
                        ('scale', StandardScaler(), scale_cols)
                    ],
                    remainder='passthrough'  # Keep other columns as is
                )

                # Select model based on choices
                if problem_type == "Regression":
                    if algorithm == "Linear":
                        model = LinearRegression()
                        file_name = "LinearRegression"
                    else:
                        model = RandomForestRegressor()
                        file_name = "RandomForestRegressor"
                else:
                    if algorithm == "Logistic":
                        model = LogisticRegression(max_iter=1000)
                        file_name = "LogisticRegression"
                    else:
                        model = RandomForestClassifier()
                        file_name = "RandomForestClassifier"

                # Full pipeline
                pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('model', model)])

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train
                pipeline.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = pipeline.predict(X_test)
                if problem_type == "Regression":
                    metric = mean_squared_error(y_test, y_pred)
                    st.write(f"Mean Squared Error: {metric}")
                else:
                    metric = accuracy_score(y_test, y_pred)
                    st.write(f"Accuracy: {metric}")

                # Save model
                print(dir(uploaded_file))
                save_model(pipeline, file_name+"_"+uploaded_file.name[:uploaded_file.name.find(".")])