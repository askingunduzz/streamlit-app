import streamlit as st
import pandas as pd
import joblib
import os

# Constructing dynamic paths to the .pkl files
model_path = os.path.join(os.path.dirname(__file__), "gradient_boosting_model.pkl")
preprocessor_path = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")

# Loading the model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


# Streamlit App Title and Description
st.title("Bank Marketing Campaign Prediction")
st.write("""
This app predicts whether a customer will subscribe to a term deposit.  
Provide customer details below to make a prediction.
""")

# Sidebar for User Input
st.sidebar.header("Enter Customer Details")

# Input fields for user data
def get_user_input():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
    job = st.sidebar.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                       "management", "retired", "self-employed", "services",
                                       "student", "technician", "unemployed", "unknown"])
    marital = st.sidebar.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
    education = st.sidebar.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.sidebar.selectbox("Default Credit?", ["yes", "no"])
    housing = st.sidebar.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.sidebar.selectbox("Personal Loan?", ["yes", "no"])
    duration = st.sidebar.number_input("Last Contact Duration (seconds)", min_value=0, step=1)
    campaign = st.sidebar.number_input("Number of Contacts in Campaign", min_value=1, step=1)

    # Add placeholder values for missing features
    data = {
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'duration': [duration],
        'campaign': [campaign],
        # Add default values for missing columns
        'cons.price.idx': [93.0],
        'day_of_week': ['mon'],
        'cons.conf.idx': [-40.0],
        'contact': ['cellular'],
        'poutcome': ['failure'],
        'pdays': [0],
        'euribor3m': [4.5],
        'month': ['may'],
        'previous': [0],
        'emp.var.rate': [1.1],
        'nr.employed': [5191.0]
    }
    return pd.DataFrame(data)

# Get user input
input_data = get_user_input()

# Display user input
st.subheader("Customer Input Data:")
st.write(input_data)

# Preprocess input data to match training pipeline
input_processed = preprocessor.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_processed)
    if prediction[0] == "yes":
        st.success("The customer is likely to subscribe to the term deposit.")
    else:
        st.error("The customer is unlikely to subscribe to the term deposit.")

# Footer
st.write("---")
st.write("*Deployed using Streamlit*")
