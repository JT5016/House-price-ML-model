import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib


# Inject custom CSS
st.markdown(
    """
    <style>
    /* Main app container: gradient background, custom font and padding */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #ADD8E6, #87CEFA);
        font-family: 'Roboto', sans-serif;
        padding: 2rem;
        color: #333;
    }
    
    /* Sidebar styling: white background with padding and a subtle border */
    [data-testid="stSidebar"] > div:first-child {
        background: #FFFFFF;
        padding: 1rem;
        border-right: 2px solid #000080;
    }
    
    /* Title styling: dark blue with a shadow and centered text */
    h1 {
        color: #000080;
        font-size: 3rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    /* General header styling (h2, h3, etc.) */
    h2, h3, h4 {
        color: #000080;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Button styling: dark blue background with rounded corners and hover effect */
    .stButton button {
        background-color: #000080;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #333399;
        transform: scale(1.02);
    }
    
    /* Input styling: add some padding and border radius to number inputs */
    .stNumberInput input {
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    
    /* Customize markdown text elements */
    .css-1d391kg p {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------
# 1. Display App Title and Description
# -----------------------------------------------------------
st.title("House Price Prediction App")
st.markdown("""
This interactive app predicts house sale prices using a Linear Regression model.
The model was trained on a dataset using the following features:
- Lot Area
- Above Ground Living Area (GrLivArea)
- Number of Garage Cars
- Number of Full Bathrooms
- Overall Quality

Enter the house features on the sidebar to get a predicted sale price.
""")

# -----------------------------------------------------------
# 2. Debug: Display Current Working Directory
# -----------------------------------------------------------


# -----------------------------------------------------------
# 3. Load the Trained Model
# -----------------------------------------------------------
try:
    model = joblib.load('ML/house_price_model.pkl')

except Exception as e:
    st.error("Error loading the model. Check the file path and try again.")
    st.error(e)

# -----------------------------------------------------------
# 4. User Input Section in Sidebar
# -----------------------------------------------------------
st.sidebar.header("Input Features")

def user_input_features():
    # Get numerical inputs from the sidebar
    lot_area = st.sidebar.number_input("Lot Area", min_value=0.0, value=7500.0)
    gr_liv_area = st.sidebar.number_input("Above Ground Living Area (GrLivArea)", min_value=0.0, value=1500.0)
    garage_cars = st.sidebar.number_input("Number of Garage Cars", min_value=0, value=2)
    full_bath = st.sidebar.number_input("Number of Full Bathrooms", min_value=0, value=2)
    overall_qual = st.sidebar.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
    
    # Build the input dictionary with the selected features
    data_input = {
        "LotArea": lot_area,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "OverallQual": overall_qual
    }
    
    # Convert the dictionary to a DataFrame (a single row)
    features_df = pd.DataFrame(data_input, index=[0])
    return features_df

# Get user inputs
input_df = user_input_features()
st.subheader("User Input Features")
st.write(input_df)

# -----------------------------------------------------------
# 5. Prediction Section
# -----------------------------------------------------------
if st.button("Predict House Price"):

    try:
        # Predict using the model (model was trained on log-transformed SalePrice)
        log_price_prediction = model.predict(input_df)
        # Convert the log prediction back to the original scale
        predicted_price = np.exp(log_price_prediction)
        st.success(f"Predicted House Sale Price: ${predicted_price[0]:,.2f}")
    except Exception as e:
        st.error("An error occurred during prediction.")
        st.error(e)
