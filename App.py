import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load model and dropdown options
model = joblib.load("model.pkl")
dropdowns = joblib.load("dropdown_options.pkl")

# Streamlit App UI
st.title("Real Estate Pricing Model")

# Build dropdowns dynamically from pre-saved options
city = st.selectbox("City:", dropdowns['city'])
product_style = st.selectbox("Product Style:", dropdowns['product_style'])
unit_size = st.number_input("Unit Size (sqft):", 500, 10000, 2500)
subdivision = st.text_input("Subdivision:", "")
seller = st.selectbox("Seller:", dropdowns['seller'])
zip_code = st.text_input("Zip Code:", "34787")

if st.button("Predict Sale Price"):
    input_df = pd.DataFrame([{
        'city': city,
        'product_style': product_style,
        'unit_size': unit_size,
        'subdivision': subdivision,
        'seller': seller,
        'zip_code': zip_code
    }])
    
    predicted_log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(predicted_log_price)

    st.subheader(f"Estimated Sale Price: ${predicted_price:,.2f}")






