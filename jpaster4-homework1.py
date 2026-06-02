import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("AutoML Application")

st.write("Upload a CSV dataset, select your target and feature variables, and compare performance of machine learning models.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Select Target and Feature Variables")
    st.write("""From the dropdown menu, select the field which you would like to predict (target variable). 
             Then, select the fields to use as model inputs for prediction of the target variable (feature variables). 
             Application includes all fields (excluding the target) as features in the models by default.""")

    numeric_fields = df.select_dtypes(include= np.number).columns.tolist()
    target = st.selectbox("Select Target Variable", numeric_fields)

    available_features = [col for col in df.columns if col != target]

    features = st.multiselect("Select Feature Variables", available_features, default=available_features, help="Remove features using the x button. Add features back by selecting them from the dropdown list.")

    if len(features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()
    
    # Subset data (optional)
    st.subheader("Subset Dataset (optional)")
    st.write("""Select the checkbox if you would like to take a subset of the dataset for faster performance. 
             Use the slider to indicate the percentage of the data you would like to use for the models.
             If you do not wish to subset the data, click the Run Models button to begin training the models.""")

    subset_data = st.checkbox("Use a subset of the data")

    if subset_data:
        subset_percent = st.slider("Percentage of dataset to use", min_value=5, max_value=100, value=75, step=5)
        if st.button("Apply Subset"):
            df = df.sample(frac=subset_percent / 100, random_state=42)
            st.success("Dataset Subsetting Complete")

    st.subheader("Regression Models")
    st.write("""Click the Run Models button to begin pre-processing the dataset for analysis and training three types of ML models: linear regression, random forest, and ridge regression.
             The results will be displayed side by side in order to compare R^2 scores, MAE, RMSE, and residuals between models.""")

    if st.button("Run Models"):

        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("Preparing dataset...")

        data = df.copy()

        # Pre-process
        data = data.dropna()

        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))

        y = data[target]
        X = data[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        progress_bar.progress(25)

        col1, col2, col3 = st.columns(3)

        with col1: # Linear
            progress_text.text("Training Linear Regression Model...")
            st.markdown("<h2 style='font-size: 20px; color: black; text-align:center;'>Linear Regression Model</h2>", unsafe_allow_html=True)
            linear_model = LinearRegression()

            linear_model.fit(X_train, y_train)
            
            # Predict and evaluate
            predictions = linear_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            st.metric("R²", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            st.metric("RMSE", f"{rmse:.3f}")

            residuals = y_test - predictions

            fig, ax = plt.subplots()

            ax.scatter(predictions, residuals, alpha=0.5)
            ax.axhline(y=0, color='red', linestyle='--')

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")

            st.pyplot(fig)

            progress_bar.progress(50)
        
        with col2: # Random Forest
            progress_text.text("Training Random Forest Model...")
            st.markdown("<h2 style='font-size: 20px; color: black; text-align:center;'>Random Forest Model</h2>", unsafe_allow_html=True)

            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            st.metric("R²", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            st.metric("RMSE", f"{rmse:.3f}")

            residuals = y_test - predictions

            fig, ax = plt.subplots()

            ax.scatter(predictions, residuals, alpha=0.5)
            ax.axhline(y=0, color='red', linestyle='--')

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")

            st.pyplot(fig)

            progress_bar.progress(75)
        
        with col3: # Ridge
            progress_text.text("Training Ridge Regression Model...")
            st.markdown("<h2 style='font-size: 20px; color: black; text-align:center;'>Ridge Regression Model</h2>", unsafe_allow_html=True)
            ridge_model = Ridge(alpha=1.0)

            ridge_model.fit(X_train, y_train)

            # Predict and evaluate
            predictions = ridge_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            st.metric("R²", f"{r2:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            st.metric("RMSE", f"{rmse:.3f}")

            residuals = y_test - predictions

            fig, ax = plt.subplots()

            ax.scatter(predictions, residuals, alpha=0.5)

            ax.axhline(y=0, color='red', linestyle='--')

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")

            st.pyplot(fig)

            progress_bar.progress(100)
    
            progress_text.success("All models completed")
