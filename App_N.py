import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load the trained model and necessary components with error handling
try:
    scaler = joblib.load("scalerN.pkl")  # Load your scaler
    pca = joblib.load("pcaN.pkl")        # Load your PCA
    classifier_model = joblib.load("classifierN.pkl")  # Load your classifier model
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Function to predict using the loaded model
def predict(features):
    if not model_loaded:
        return "Model not loaded"

    # Create a DataFrame from the input features
    features_df = pd.DataFrame([features], columns=[
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 
        'ConvexArea', 'EquivDiameter', 'Extent', 'Perimeter', 
        'Roundness', 'AspectRation'
    ])

    # Scale the features
    scaled_features = scaler.transform(features_df)
    # Apply PCA
    pca_features = pca.transform(scaled_features)

    # Predict using the best model
    prediction = classifier_model.predict(pca_features)
    return prediction[0]

def main():
    # Display an image at the top (optional)
    st.image("Rice.jpg", width=700)  # Replace with your image path if needed

    # URL for the background image
    background_image_url = "https://img.freepik.com/premium-photo/organic-rice-mixed-rice-backdrop-texture_872147-13487.jpg"

    # Custom CSS for background image and input styles
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        label {{
            font-size: 24px !important;
            font-weight: bold;
            color: black !important;
            display: block;
            padding: 10px;
            border-radius: 10px;
            background-color: #F5F5DC;
            color: #fff;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # HTML for styling
    html_temp = """
    <div style="background-color:#5C4033;padding:10px">
    <h2 style="color:white;text-align:center;">Rice Type Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create input fields for each feature with validation
    Area = st.number_input("Area", min_value=0.0, format="%.2f")
    MajorAxisLength = st.number_input("Major Axis Length", min_value=0.0, format="%.2f")
    MinorAxisLength = st.number_input("Minor Axis Length", min_value=0.0, format="%.2f")
    Eccentricity = st.number_input("Eccentricity", min_value=0.0, format="%.2f")
    ConvexArea = st.number_input("Convex Area", min_value=0.0, format="%.2f")
    EquivDiameter = st.number_input("Equivalent Diameter", min_value=0.0, format="%.2f")
    Extent = st.number_input("Extent", min_value=0.0, format="%.2f")
    Perimeter = st.number_input("Perimeter", min_value=0.0, format="%.2f")
    Roundness = st.number_input("Roundness", min_value=0.0, format="%.2f")
    AspectRation = st.number_input("Aspect Ratio", min_value=0.0, format="%.2f")

    # Gather inputs into a list in the correct order
    features = [
        Area, MajorAxisLength, MinorAxisLength, Eccentricity, 
        ConvexArea, EquivDiameter, Extent, Perimeter, 
        Roundness, AspectRation
    ]

    # When 'Predict' button is clicked
    if st.button("Predict"):
        if model_loaded:
            result = predict(features)
            if result != "Model not loaded":
                # Display the prediction result in a styled box with green font color
                st.markdown(
                    f"""
                    <div style="border: 2px solid green; padding: 10px; border-radius: 5px; background-color: #eaffea;">
                        <h3 style="color: green; text-align: center;">The predicted class is: {result}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Prediction failed. Model not loaded properly.")
        else:
            st.error("Model is not loaded. Please check the model files.")

if __name__ == '__main__':
    main()
