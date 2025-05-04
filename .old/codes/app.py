import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest  # Example model, replace if using a different one

st.markdown("<h1 style='text-align: center; font-size: 48px; color: red;'>Credit Card Fraud Anomaly Detection ML App</h1>", unsafe_allow_html=True)

# Load the credit card dataset directly
@st.cache_data
def load_dataset():
    return pd.read_csv('creditcard.csv')  # Path to the creditcard.csv file

# Cache function to convert DataFrame to CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

# Load pre-trained anomaly detection model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load the dataset
df = load_dataset()

# Display the dataset preview
st.write("Dataset Preview:")
st.dataframe(df.head())

# Ensure the dataset has the 'Class' column
if 'Class' not in df.columns:
    st.error("The dataset must contain a 'Class' column.")
else:
    # Add functionality for row selection
    st.markdown("### Select a Row for Model Input:")

    # Option to select a row
    selected_row = st.selectbox("Select a Row", options=range(len(df)), index=0)

    # Add a button below the row selection for anomaly detection
    detect_button = st.button("Detect Anomaly")

    # If the "Detect" button is clicked
    if detect_button:
        # Row to use for the model
        row_to_use = df.iloc[selected_row]

        # Drop 'Class' column (exclude it from features used for prediction)
        row_to_use_for_model = row_to_use.drop('Class', errors='ignore')

        # Check if the number of features matches the model's expectations
        model = load_model()

        if len(row_to_use_for_model) != model.n_features_in_:
            st.error(f"The model expects {model.n_features_in_} features, but {len(row_to_use_for_model)} were provided.")
        else:
            # Apply the model for anomaly detection
            prediction = model.predict([row_to_use_for_model])

            # Display the row and the anomaly detection result
            st.write(f"Row selected for anomaly detection:")
            st.write(row_to_use)

            # Show the anomaly result for the selected row
            result = "Anomaly" if prediction[0] == -1 else "Not Anomaly"
            st.write(f"Anomaly Detection Result: {result}")

            # Provide option to download the result
            result_df = row_to_use.to_frame().T  # Convert Series to DataFrame
            result_df['Anomaly'] = result
            result_csv = convert_df(result_df)
            st.download_button(
                label="Download Selected Row Result",
                data=result_csv,
                file_name="Selected_Row_Anomaly_Result.csv",
                mime="text/csv",
            )
