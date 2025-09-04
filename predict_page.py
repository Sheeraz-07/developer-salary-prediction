import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_model():
    try:
        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def show_predict_page():
    st.title("DEVELOPER SALARY PREDICTOR")
    st.write("""### We need some information to predict salary""")
    
    try:
        data = load_model()
        regressor_loaded = data["model"]
        le_country = data["le_country"]
        le_education = data["le_education"]
    except Exception as e:
        st.error(f"Failed to load model components: {str(e)}")
        return

    countries = (
        "United States",
        "India", 
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden"
    )
    
    education = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree", 
        "Post grad"
    )
    
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", education)
    experience = st.slider("Years of Experience", 0, 50, 3)
    
    if st.button("Predict Salary"):
        try:
            X_df = pd.DataFrame({
                'Country': [country],
                'EdLevel': [education], 
                'YearsCodePro': [experience]
            })
            
            X_df['Country'] = le_country.transform(X_df['Country'])
            X_df['EdLevel'] = le_education.transform(X_df['EdLevel'])
            
            salary = regressor_loaded.predict(X_df)
            st.subheader(f"The estimated Salary is: ${salary[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")