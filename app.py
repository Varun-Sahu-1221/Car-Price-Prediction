import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Car Value Predictor",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_models():
    ridge = joblib.load('ridge_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return ridge, scaler

model, scaler = load_models()

expected_columns = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 
    'enginesize', 'horsepower', 'citympg', 'carbody_hardtop', 
    'carbody_hatchback', 'carbody_sedan', 'carbody_wagon', 'drivewheel_fwd', 
    'drivewheel_rwd', 'enginelocation_rear', 'enginetype_dohcv', 
    'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf', 'enginetype_ohcv', 
    'enginetype_rotor', 'cylindernumber_five', 'cylindernumber_four', 
    'cylindernumber_six', 'cylindernumber_three', 'cylindernumber_twelve', 
    'cylindernumber_two'
]

st.title("🏎️ Car Value Predictor")
st.markdown("Enter the specifications of the vehicle below.")


with st.form("car_specs_form"):
    
    st.subheader("Numeric Specifications")
    col1, col2 = st.columns(2, gap="large") 
    
    with col1:
        symboling = st.number_input("Insurance Risk Rating (-3 to 3)", min_value=-3, max_value=3, value=0, help="Higher means riskier")
        wheelbase = st.number_input("Wheelbase (inches)", min_value=80.0, max_value=130.0, value=98.0, step=0.1)
        carlength = st.number_input("Car Length (inches)", min_value=140.0, max_value=210.0, value=174.0, step=0.1)
        carwidth = st.number_input("Car Width (inches)", min_value=60.0, max_value=75.0, value=65.9, step=0.1)
        
    with col2:
        curbweight = st.number_input("Curb Weight (lbs)", min_value=1400, max_value=4200, value=2500, step=50)
        enginesize = st.number_input("Engine Size (cc)", min_value=60, max_value=330, value=120, step=10)
        horsepower = st.number_input("Horsepower (hp)", min_value=40, max_value=300, value=100, step=5)
        citympg = st.number_input("City MPG", min_value=10, max_value=60, value=25, step=1)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Categorical Specifications")
    cat_col1, cat_col2, cat_col3 = st.columns(3, gap="medium")
    
    with cat_col1:
        carbody = st.selectbox("Car Body Style", ["convertible", "hardtop", "hatchback", "sedan", "wagon"])
        drivewheel = st.selectbox("Drive Wheel", ["4wd", "fwd", "rwd"])

    with cat_col2:
        enginelocation = st.selectbox("Engine Location", ["front", "rear"])
        enginetype = st.selectbox("Engine Type", ["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"])

    with cat_col3:
        cylindernumber = st.selectbox("Number of Cylinders", ["two", "three", "four", "five", "six", "eight", "twelve"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    submit_button = st.form_submit_button("Calculate Estimated Value", type="primary", use_container_width=True)

if submit_button:
    input_data = {col: 0 for col in expected_columns}
    
    input_data.update({
        'symboling': symboling, 'wheelbase': wheelbase, 'carlength': carlength,
        'carwidth': carwidth, 'curbweight': curbweight, 'enginesize': enginesize,
        'horsepower': horsepower, 'citympg': citympg
    })
    
    if f'carbody_{carbody}' in expected_columns: input_data[f'carbody_{carbody}'] = 1
    if f'drivewheel_{drivewheel}' in expected_columns: input_data[f'drivewheel_{drivewheel}'] = 1
    if f'enginelocation_{enginelocation}' in expected_columns: input_data[f'enginelocation_{enginelocation}'] = 1
    if f'enginetype_{enginetype}' in expected_columns: input_data[f'enginetype_{enginetype}'] = 1
    if f'cylindernumber_{cylindernumber}' in expected_columns: input_data[f'cylindernumber_{cylindernumber}'] = 1
    
    df_input = pd.DataFrame([input_data])
    
    try:
        scaled_features = scaler.transform(df_input)
        prediction = model.predict(scaled_features)
        
        st.divider()
        st.subheader("Prediction Result")
        final_price = max(0, prediction[0]) 
        st.metric(label="Estimated Market Value", value=f"${final_price:,.2f}")
        
    except Exception as e:
        st.error(f"Prediction failed. Error details: {e}")