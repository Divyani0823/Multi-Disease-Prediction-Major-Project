import streamlit as st
import numpy as np
import pickle

# Load the models
diabetes_model = pickle.load(open(r'models\diabetes.pkl', 'rb'))
cancer_model = pickle.load(open(r'models\cancer.pkl', 'rb'))
heart_model = pickle.load(open(r'models\heart.pkl', 'rb'))

# Function to predict Diabetes
def predict_diabetes(data):
    prediction = diabetes_model.predict(data)
    return prediction

# Function to predict Heart Disease
def predict_heart(data):
    prediction = heart_model.predict(data)
    return prediction

# Function to predict Cancer
def predict_cancer(data):
    prediction = cancer_model.predict(data)
    return prediction

# Streamlit UI
st.title("Multi Disease Prediction System")

# Sidebar navigation
option = st.sidebar.selectbox(
    'Select Disease for Prediction',
    ['Diabetes', 'Heart Disease', 'Cancer']
)

# Diabetes Prediction
if option == 'Diabetes':
    st.header("Diabetes Prediction")
    preg = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
    bp = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)
    st = st.number_input('Skin Thickness', min_value=0, max_value=50, value=0)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
    age = st.number_input('Age', min_value=0, max_value=100, value=0)
    
    data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

    if st.button('Predict Diabetes'):
        prediction = predict_diabetes(data)
        st.write("Prediction: ", "Diabetic" if prediction == 1 else "Non-Diabetic")

# Heart Disease Prediction
elif option == 'Heart Disease':
    st.header("Heart Disease Prediction")
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [1, 2, 3])

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if st.button('Predict Heart Disease'):
        prediction = predict_heart(data)
        st.write("Prediction: ", "Heart Disease Present" if prediction == 1 else "No Heart Disease")

# Cancer Prediction
elif option == 'Cancer':
    st.header("Cancer Prediction")
    radius_mean = st.number_input('Radius Mean', min_value=0.0, max_value=100.0, value=0.0)
    texture_mean = st.number_input('Texture Mean', min_value=0.0, max_value=100.0, value=0.0)
    perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, max_value=200.0, value=0.0)
    area_mean = st.number_input('Area Mean', min_value=0.0, max_value=2000.0, value=0.0)
    smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=0.1, value=0.0)
    compactness_mean = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, value=0.0)
    concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=1.0, value=0.0)
    concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, max_value=1.0, value=0.0)
    symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, max_value=1.0, value=0.0)
    radius_se = st.number_input('Radius SE', min_value=0.0, max_value=10.0, value=0.0)
    perimeter_se = st.number_input('Perimeter SE', min_value=0.0, max_value=100.0, value=0.0)
    area_se = st.number_input('Area SE', min_value=0.0, max_value=500.0, value=0.0)
    compactness_se = st.number_input('Compactness SE', min_value=0.0, max_value=1.0, value=0.0)
    concavity_se = st.number_input('Concavity SE', min_value=0.0, max_value=1.0, value=0.0)
    concave_points_se = st.number_input('Concave Points SE', min_value=0.0, max_value=0.1, value=0.0)
    fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0, max_value=1.0, value=0.0)
    radius_worst = st.number_input('Radius Worst', min_value=0.0, max_value=100.0, value=0.0)
    texture_worst = st.number_input('Texture Worst', min_value=0.0, max_value=100.0, value=0.0)
    perimeter_worst = st.number_input('Perimeter Worst', min_value=0.0, max_value=200.0, value=0.0)
    area_worst = st.number_input('Area Worst', min_value=0.0, max_value=2000.0, value=0.0)
    smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0, max_value=0.1, value=0.0)
    compactness_worst = st.number_input('Compactness Worst', min_value=0.0, max_value=1.0, value=0.0)
    concavity_worst = st.number_input('Concavity Worst', min_value=0.0, max_value=1.0, value=0.0)
    concave_points_worst = st.number_input('Concave Points Worst', min_value=0.0, max_value=0.1, value=0.0)
    symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0, max_value=1.0, value=0.0)
    fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.0, max_value=1.0, value=0.0)

    data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                      radius_se, perimeter_se, area_se, compactness_se, concavity_se, concave_points_se, fractal_dimension_se, radius_worst, texture_worst,
                      perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])

    if st.button('Predict Cancer'):
        prediction = predict_cancer(data)
        st.write("Prediction: ", "Cancer Present" if prediction == 1 else "No Cancer")

