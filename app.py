import numpy as np 
import pandas as pd 
import streamlit as st
import pickle

model=pickle.load(open(r'/Users/puravdoshi/Downloads/CarPricePredictor/LinearRegressionModel.pkl','rb'))
final_df=pd.read_csv(r'/Users/puravdoshi/Downloads/CarPricePredictor/clean_car.csv')

# Page title
st.title("🚗 Car Price Predictor")

# Dropdowns
name = st.selectbox("Select Car Name", sorted(final_df['name'].unique()))
company = st.selectbox("Select Company", sorted(final_df['company'].unique()))
year = st.selectbox("Select Year", sorted(final_df['year'].unique(), reverse=True))
kms_driven = st.number_input("Enter Kilometers Driven", min_value=0, max_value=1000000, step=500)
fuel_type = st.selectbox("Select Fuel Type", final_df['fuel_type'].unique())

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5))
    # Predict
    predicted_price = model.predict(input_df)[0]
    # Show result
    st.success(f"Estimated Car Price: ₹ {int(predicted_price):,}")