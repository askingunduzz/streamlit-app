import streamlit as st
import pandas as pd
import joblib

# Load the model and preprocessor
model = joblib.load('gradient_boosting_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Bank Marketing Prediction App")
st.write("This app predicts whether a customer will subscribe to a term deposit.")

# Create input fields for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    job = st.sidebar.selectbox('Job', ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'housemaid', 'entrepreneur', 'student', 'self-employed', 'unknown'])
    marital = st.sidebar.selectbox('Marital Status', ['married', 'single', 'divorced'])
    education = st.sidebar.selectbox('Education', ['university.degree', 'high.school', 'basic.9y', 'basic.4y', 'basic.6y', 'professional.course', 'unknown'])
    default = st.sidebar.selectbox('Default Credit?', ['yes', 'no'])
    housing = st.sidebar.selectbox('Housing Loan?', ['yes', 'no'])
    loan = st.sidebar.selectbox('Personal Loan?', ['yes', 'no'])
    contact = st.sidebar.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    month = st.sidebar.selectbox('Month of Last Contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.sidebar.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    campaign = st.sidebar.number_input('Number of Contacts', min_value=1, value=1)
    
    # Create DataFrame from inputs
    data = {
        'age': age, 
        'job': job, 
        'marital': marital, 
        'education': education,
        'default': default, 
        'housing': housing, 
        'loan': loan, 
        'contact': contact, 
        'month': month, 
        'day_of_week': day_of_week,
        'campaign': campaign
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

# Ensure input_df has all expected columns
missing_cols = set(preprocessor.feature_names_in_) - set(input_df.columns)
for col in missing_cols:
    if col in preprocessor.transformers_[0][2]:  # Numerical columns
        input_df[col] = 0  # Default for missing numerical features
    else:  # Categorical columns
        input_df[col] = "unknown"  # Default for missing categorical features

# Align input_df to match training data columns
input_df = input_df[preprocessor.feature_names_in_]

# Preprocess inputs and make predictions
processed_input = preprocessor.transform(input_df)
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

st.subheader("Prediction")
st.write("Yes" if prediction[0] == 1 else "No")

st.subheader("Prediction Probability")
st.write(prediction_proba)

# Deploy and test again
