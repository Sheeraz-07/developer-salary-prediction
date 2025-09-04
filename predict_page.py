import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

@st.cache_data
def load_model():
    try:
        # Import sklearn modules inside the function
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.preprocessing import LabelEncoder
        
        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except ImportError as e:
        st.error(f"Required module not found: {str(e)}. Please check requirements.txt")
        st.stop()
    except FileNotFoundError:
        st.error("Model file 'saved_steps.pkl' not found. Please ensure the file is in the repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def show_predict_page():
    st.title("DEVELOPER SALARY PREDICTOR")
    st.write("""### We need some information to predict salary""")
    
    # Load model data inside the function
    data = load_model()
    regressor_loaded = data["model"]
    le_country = data["le_country"]
    le_education = data["le_education"]

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
    ok = st.button("Predict Salary")
    if ok:
        # Create DataFrame with proper feature names
        X_df = pd.DataFrame({
            'Country': [country],
            'EdLevel': [education],
            'YearsCodePro': [experience]
        })
        
        # Transform categorical variables
        X_df['Country'] = le_country.transform(X_df['Country'])
        X_df['EdLevel'] = le_education.transform(X_df['EdLevel'])
        
        salary = regressor_loaded.predict(X_df)
        st.subheader(f"The estimated Salary is: ${salary[0]:.2f}")