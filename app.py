import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Caching the loading process so it doesn't reload on every interaction
@st.cache(allow_output_mutation=True)
def load_model_and_preprocessors():
    # Load the trained model (ensure the file is in the same folder or adjust the path)
    with open('Tens_orientation.pkl', 'rb') as file:
        model = pickle.load(file)
        
    # Load the training data used for preprocessing (adjust the path if needed)
    df = pd.read_csv("train1.csv")
    
    # Fit the OneHotEncoder on the categorical features
    encoder = OneHotEncoder()
    df_encoded = encoder.fit_transform(df[['orientation', 'infill_pattern']])
    odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
    
    # Select the numeric features and target as done during training
    idf = df[['layer_thick', 'infill_density', 'mwcnt', 'graphene', 'tensile_str']]
    
    # Combine encoded categorical data with numeric data to mimic training
    cdf = pd.concat([odf, idf], axis=1)
    cdf.columns = cdf.columns.astype(str)
    
    # Fit a StandardScaler on the entire DataFrame (including the target column)
    scaler = StandardScaler()
    scaler.fit(cdf)
    
    # Separately, fit a scaler on the target so we can inverse transform the prediction
    scaler_y = StandardScaler()
    scaler_y.fit(df[['tensile_str']])
    
    return model, encoder, scaler, scaler_y

# Load model and preprocessing objects
model, encoder, scaler, scaler_y = load_model_and_preprocessors()

st.title("Tensile Strength Prediction App")
st.write("Enter the feature values below to predict the tensile strength of your material.")

# Create input widgets for user features
st.header("Input Features")

# For categorical features, extract available categories from the fitted encoder
orientation_options = encoder.categories_[0].tolist()
infill_options = encoder.categories_[1].tolist()

orientation = st.selectbox("Orientation", orientation_options)
infill_pattern = st.selectbox("Infill Pattern", infill_options)

layer_thick = st.number_input("Layer Thickness", min_value=0.0, value=0.1, step=0.01)
infill_density = st.number_input("Infill Density", min_value=0.0, value=20.0, step=0.1)
mwcnt = st.number_input("MWCNT", min_value=0.0, value=0.5, step=0.1)
graphene = st.number_input("Graphene", min_value=0.0, value=0.5, step=0.1)

# When the user clicks the prediction button
if st.button("Predict Tensile Strength"):
    # Create a DataFrame with the user input
    input_df = pd.DataFrame({
        "orientation": [orientation],
        "infill_pattern": [infill_pattern],
        "layer_thick": [layer_thick],
        "infill_density": [infill_density],
        "mwcnt": [mwcnt],
        "graphene": [graphene]
    })
    
    # Process the categorical variables using the fitted OneHotEncoder
    input_encoded = encoder.transform(input_df[['orientation', 'infill_pattern']])
    input_odf = pd.DataFrame.sparse.from_spmatrix(input_encoded)
    
    # Get the numeric features
    input_numeric = input_df[['layer_thick', 'infill_density', 'mwcnt', 'graphene']]
    
    # Combine the encoded categorical features with numeric features
    input_combined = pd.concat([input_odf, input_numeric], axis=1)
    input_combined.columns = input_combined.columns.astype(str)
    
    # IMPORTANT:
    # The scaler was originally fitted on a DataFrame that included the target 'tensile_str'.
    # To mimic that exactly, we add a dummy column for 'tensile_str' (value doesn't matter)
    input_combined['tensile_str'] = 0
    
    # Apply the same StandardScaler used during training
    scaled_input = scaler.transform(input_combined)
    
    # Remove the dummy target column (assumed to be the last column) to get the features only
    scaled_input = scaled_input[:, :-1]
    
    # Use the loaded model to make a prediction (note that the model was trained on scaled data)
    pred_scaled = model.predict(scaled_input)
    
    # Inverse transform the scaled prediction to get it back to the original scale
    pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    st.success(f"Predicted Tensile Strength: {pred_original[0,0]:.2f}")
