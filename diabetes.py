import streamlit as st
import pickle
import numpy as np

# Load the pre-trained Gradient Boosting model
with open('gb_model.pkl', 'rb') as file:
    loaded_gb_model = pickle.load(file)

# Streamlit app
def main():
    st.title("Diabetes Prediction App")
    st.sidebar.header("User Input Features")

    # Create input fields for user to enter data
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin", 0, 846, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.3725)
    age = st.sidebar.slider("Age", 21, 81, 29)

    # Create a feature vector from user inputs
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Make predictions using the loaded model
    prediction = loaded_gb_model.predict(input_data)[0]
    prediction_proba = loaded_gb_model.predict_proba(input_data)[0][1]

    st.sidebar.markdown("### Prediction")
    st.sidebar.write("0: No Diabetes, 1: Diabetes")
    st.sidebar.write(f"Prediction: {prediction} (Probability: {prediction_proba:.2f})")

if __name__ == '__main__':
    main()
