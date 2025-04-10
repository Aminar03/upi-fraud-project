import os
import base64
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
from xgboost import XGBClassifier

# ------------------------- #
# Load Model with Error Handling
# ------------------------- #
MODEL_PATH = r"C:\Users\amina\Downloads\UPI Fraud Detection Final (2).pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"🚨 Model file not found! Expected at: {MODEL_PATH}")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ------------------------- #
# Define Categorical Options
# ------------------------- #
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
      'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
      'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim',
      'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment',
      'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

# ------------------------- #
# UI Header
# ------------------------- #
st.title("🔍 UPI Transaction Fraud Detector")
st.markdown(
    """
    **Check if a UPI transaction is fraudulent!**
    - You can manually enter transaction details **or upload a CSV file** for batch processing.
    """
)

# ------------------------- #
# User Inputs for Single Transaction
# ------------------------- #
st.subheader("📌 Enter Transaction Details")

tran_date = st.date_input("📅 Select transaction date", datetime.date.today())
selected_date = dt.combine(tran_date, dt.min.time())
month, year = selected_date.month, selected_date.year

tran_type = st.selectbox("📌 Select transaction type", tt)
pmt_gateway = st.selectbox("💰 Select payment gateway", pg)
tran_state = st.selectbox("📍 Select transaction state", ts)
merch_cat = st.selectbox("🏪 Select merchant category", mc)
amt = st.number_input("💵 Enter transaction amount", min_value=0.0, step=0.1)

st.write("OR")

# ------------------------- #
# File Upload for Bulk Transactions
# ------------------------- #
st.subheader("📂 Upload a CSV File")

SAMPLE_CSV_PATH = "sample.csv"
if os.path.exists(SAMPLE_CSV_PATH):
    df_sample = pd.read_csv(SAMPLE_CSV_PATH)
    st.write("✅ Example CSV Format:", df_sample.head())
else:
    st.warning("⚠️ Sample CSV file not found! Please ensure it is in the same folder.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
df_uploaded = None
if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded CSV Preview:", df_uploaded.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

# ------------------------- #
# Predict Fraud
# ------------------------- #
if st.button("🔍 Check Transaction(s)"):
    if df_uploaded is not None:
        with st.spinner("Checking transactions..."):
            def download_csv(df):
                csv = df.to_csv(index=False, header=True)
                b64 = base64.b64encode(csv.encode()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="output.csv">📥 Download Output CSV</a>'

            # Convert Date to Year & Month
            df_uploaded[['Month', 'Year']] = df_uploaded['Date'].str.split('-', expand=True)[[1, 2]]
            df_uploaded[['Month', 'Year']] = df_uploaded[['Month', 'Year']].astype(int)
            df_uploaded.drop(columns=['Date'], inplace=True)

            # Reorder columns for the model
            df_uploaded = df_uploaded.reindex(columns=['Amount', 'Year', 'Month',
                                                        'Transaction_Type', 'Payment_Gateway',
                                                        'Transaction_State', 'Merchant_Category'])

            results = []
            for _, row in df_uploaded.iterrows():
                # One-hot encoding
                input_vector = [row.Amount, row.Year, row.Month]
                input_vector += [1 if row.Transaction_Type == t else 0 for t in tt]
                input_vector += [1 if row.Payment_Gateway == p else 0 for p in pg]
                input_vector += [1 if row.Transaction_State == s else 0 for s in ts]
                input_vector += [1 if row.Merchant_Category == m else 0 for m in mc]

                prediction = loaded_model.predict([input_vector])[0]
                results.append(prediction)

            df_uploaded['Fraud'] = results
            st.success("✅ Transaction check complete!")
            st.markdown(download_csv(df_uploaded), unsafe_allow_html=True)

    else:
        with st.spinner("Checking transaction..."):
            # One-hot encoding for single transaction
            input_vector = [amt, year, month]
            input_vector += [1 if tran_type == t else 0 for t in tt]
            input_vector += [1 if pmt_gateway == p else 0 for p in pg]
            input_vector += [1 if tran_state == s else 0 for s in ts]
            input_vector += [1 if merch_cat == m else 0 for m in mc]

            prediction = loaded_model.predict([input_vector])[0]

            st.success("✅ Transaction check complete!")
            if prediction == 0:
                st.success("🎉 Safe Transaction! This is NOT fraudulent.")
            else:
                st.error("🚨 Warning! This transaction is FRAUDULENT.")

