import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np

@st.cache_resource
def train_model():
    df = pd.read_csv('loan_data.csv')
    df_model = pd.get_dummies(df, columns=['purpose'], drop_first=True)
    X = df_model.drop('not.fully.paid', axis=1)
    y = df_model['not.fully.paid']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_resampled, y_resampled)
    return model

model = train_model()
threshold = 0.14

st.title("Loan Default Prediction")
st.write("Enter borrower details to predict likelihood of loan default.")

credit_policy = st.selectbox("Meets Credit Policy?", [1, 0])
purpose = st.selectbox("Loan Purpose", [
    'credit_card', 'debt_consolidation', 'educational',
    'home_improvement', 'major_purchase', 'small_business', 'all_other'
])
int_rate = st.slider("Interest Rate", 0.06, 0.22, 0.12)
installment = st.number_input("Monthly Installment ($)", 15.0, 1000.0, 300.0)
log_annual_inc = st.number_input("Log Annual Income", 7.5, 15.0, 10.9)
dti = st.number_input("Debt-to-Income Ratio", 0.0, 30.0, 12.0)
fico = st.slider("FICO Score", 612, 827, 710)
days_cr_line = st.number_input("Days with Credit Line", 0.0, 20000.0, 4500.0)
revol_bal = st.number_input("Revolving Balance ($)", 0, 200000, 15000)
revol_util = st.number_input("Revolving Utilization (%)", 0.0, 100.0, 50.0)
inq_last_6mths = st.number_input("Credit Inquiries (Last 6 Months)", 0, 10, 1)
delinq_2yrs = st.number_input("Delinquencies (Last 2 Years)", 0, 10, 0)
pub_rec = st.number_input("Public Records", 0, 5, 0)

purpose_cols = {
    'purpose_credit_card': 0, 'purpose_debt_consolidation': 0,
    'purpose_educational': 0, 'purpose_home_improvement': 0,
    'purpose_major_purchase': 0, 'purpose_small_business': 0
}
if purpose != 'all_other':
    key = f'purpose_{purpose}'
    if key in purpose_cols:
        purpose_cols[key] = 1

if st.button("Predict"):
    features = np.array([[
        credit_policy, int_rate, installment, log_annual_inc,
        dti, fico, days_cr_line, revol_bal, revol_util,
        inq_last_6mths, delinq_2yrs, pub_rec,
        purpose_cols['purpose_credit_card'],
        purpose_cols['purpose_debt_consolidation'],
        purpose_cols['purpose_educational'],
        purpose_cols['purpose_home_improvement'],
        purpose_cols['purpose_major_purchase'],
        purpose_cols['purpose_small_business']
    ]])
    
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= threshold)
    
    st.subheader("Result")
    st.write(f"Default Probability: {prob:.2%}")
    
    if prediction == 1:
        st.error("HIGH RISK - Likely to Default")
    else:
        st.success("LOW RISK - Likely to Repay")
    
    st.info(f"Model flags loans with default probability >= {threshold:.0%}")
