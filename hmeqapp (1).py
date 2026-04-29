
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #f0f8ff; padding: 20px; color: #4682b4; border-radius: 15px; font-family: \"'Roboto'\", sans-serif; box-shadow: 3px 3px 10px rgba(0,0,0,0.2);'>🏦💰 Loan Approval Predictor 📈📊</h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numerical values based on the model's features
fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=700, step=1, help="Enter the applicant's credit score (300-850).")
granted_loan_amount = st.number_input("Granted Loan Amount", min_value=5000, max_value=100000, value=25000, step=1000, help="The amount of loan granted.")
monthly_gross_income = st.number_input("Monthly Gross Income", min_value=500, max_value=20000, value=5000, step=100, help="Applicant's total monthly income before deductions.")
monthly_housing_payment = st.number_input("Monthly Housing Payment", min_value=100, max_value=10000, value=1500, step=50, help="Applicant's total monthly housing expenses.")
ever_bankrupt_or_foreclose = st.radio("Ever Bankrupt or Foreclose", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True, help="Has the applicant ever been bankrupt or foreclosed?")

# Categorical inputs with options based on the model's features
reason = st.selectbox("Reason for Loan", [
    'credit_card_refinancing', 'debt_conslidation', 'home_improvement',
    'major_purchase', 'cover_an_unexpected_cost', 'other'
], help="The primary reason for requesting the loan.")
employment_status = st.selectbox("Employment Status", ['full_time', 'unemployed', 'part_time', 'self_employed'], help="Applicant's current employment status.")
employment_sector = st.selectbox("Employment Sector", [
    'industrials', 'other', 'real_estate', 'financials', 'consumer_staples',
    'consumer_discretionary', 'telecommunication_services', 'materials',
    'health_care', 'utilities', 'information_technology', 'Unknown', 'energy'
], help="The industry sector of the applicant's employer.")
lender = st.selectbox("Lender", ['A', 'B', 'C'], help="The specific lender offering the loan.")

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "FICO_score": [fico_score],
    "Granted_Loan_Amount": [granted_loan_amount],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_or_foreclose],
    "Reason": [reason],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
categorical_cols = ['Reason', 'Employment_Status', 'Employment_Sector', 'Lender']
input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):    
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]
    prediction_proba = model.predict_proba(input_data_encoded)[:, 1][0]

    # Display result
    if prediction == 1:
        st.success(f"✅ The model predicts: **Approved!** (Probability: {prediction_proba:.2f})")
    else:
        st.error(f"❌ The model predicts: **Denied.** (Probability: {prediction_proba:.2f})")
