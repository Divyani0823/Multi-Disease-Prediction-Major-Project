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
st.set_page_config(page_title="🩺 Illness Insight ", page_icon="⚕️", layout="wide")

# Enhanced Title with medical color scheme and relevant symbols
st.markdown(
    """
    <div style="background-color:#AEDFF7;padding:10px;border-radius:10px">
    <h1 style="color:#0F4C81;text-align:center">🩺 Illness Insight </h1>
    <h3 style="color:#333333;text-align:center">Your Reliable Health Prediction Companion ⚕️</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Sidebar navigation
st.sidebar.title("🩺 Choose the Disease")
option = st.sidebar.radio(
    '🔍 Select Disease for Prediction',
    ['Diabetes', 'Heart Disease', 'Cancer']
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #F0F8FF;
    }
    .sidebar .sidebar-content {
        background: #E3F2FD;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Diabetes Prediction
if option == 'Diabetes':
    st.header("🩸 Diabetes Prediction")
    with st.form(key='diabetes_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            preg = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, help="Number of times pregnant")
            glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0, help="Plasma glucose concentration")
            bp = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0, help="Diastolic blood pressure (mm Hg)")
            skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=50, value=0, help="Triceps skin fold thickness (mm)")

        with col2:
            insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0, help="2-Hour serum insulin (mu U/ml)")
            bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0, help="Body Mass Index (weight in kg/(height in m)^2)")
            dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0, help="Diabetes pedigree function")
            age = st.number_input('Age', min_value=0, max_value=100, value=0, help="Age in years")

        data = np.array([[preg, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

        submit = st.form_submit_button('🔍 Predict Diabetes')
        if submit:
            with st.spinner('Analyzing...'):
                prediction = predict_diabetes(data)
                if prediction == 1:
                    st.error("⚠️ **Prediction: Diabetic**")
                    st.markdown(
                        """
                        **Suggestion:** 
                        - Maintain a healthy diet.
                        - Exercise regularly.
                        - Monitor blood sugar levels frequently.
                        - Consult with a healthcare professional for a management plan.
                        """
                    )
                else:
                    st.success("🎉 **Prediction: Non-Diabetic**")
                    st.markdown(
                        """
                        **Suggestion:** 
                        - Continue with a balanced diet.
                        - Regular check-ups to monitor blood sugar levels.
                        """
                    )

# Heart Disease Prediction
elif option == 'Heart Disease':
    st.header("❤️ Heart Disease Prediction")
    with st.form(key='heart_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=0, max_value=120, value=0, help="Age in years")
            sex = st.selectbox('Sex', ['Male', 'Female'], help="Gender of the patient")
            sex = 1 if sex == 'Male' else 0
            cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], help="Chest pain type (0-3)")
            trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120, help="Resting blood pressure (mm Hg)")
            chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200, help="Serum cholesterol in mg/dl")
        
        with col2:
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], help="1 if true; 0 if false")
            restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], help="Resting ECG results (0-2)")
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=220, value=150, help="Maximum heart rate achieved")
            exang = st.selectbox('Exercise Induced Angina', [0, 1], help="1 if true; 0 if false")
            oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0, help="ST depression induced by exercise relative to rest")
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2], help="Slope of the peak exercise ST segment (0-2)")
            ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3], help="Number of major vessels (0-3)")
            thal = st.selectbox('Thalassemia', [1, 2, 3], help="Thalassemia (1-3)")

        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        submit = st.form_submit_button('🔍 Predict Heart Disease')
        if submit:
            with st.spinner('Analyzing...'):
                prediction = predict_heart(data)
                if prediction == 1:
                    st.error("⚠️ **Prediction: Heart Disease Present**")
                    st.markdown(
                        """
                        **Suggestion:** 
                        - Maintain a heart-healthy diet (low in sodium, saturated fats, and cholesterol).
                        - Regular cardiovascular exercises.
                        - Routine medical check-ups with a cardiologist.
                        - Monitor blood pressure and cholesterol levels.
                        """
                    )
                else:
                    st.success("🎉 **Prediction: No Heart Disease**")
                    st.markdown(
                        """
                        **Suggestion:** 
                        - Keep up a balanced diet.
                        - Regular physical activities.
                        - Monitor heart health through yearly check-ups.
                        """
                    )
# Cancer Prediction
elif option == 'Cancer':
    st.header("🎗️ Cancer Prediction")
    with st.form(key='cancer_form'):
        st.markdown("### Input Tumor Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_mean = st.number_input('Radius Mean', min_value=0.0, max_value=100.0, value=0.0, help="Mean of distances from center to points on the perimeter")
            texture_mean = st.number_input('Texture Mean', min_value=0.0, max_value=100.0, value=0.0, help="Standard deviation of gray-scale values")
            perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, max_value=200.0, value=0.0, help="Mean size of the tumor perimeter")
            area_mean = st.number_input('Area Mean', min_value=0.0, max_value=2000.0, value=0.0, help="Mean area of the tumor")
            smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=0.1, value=0.0, help="Mean of local variation in radius lengths")
            compactness_mean = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, value=0.0, help="Mean of (perimeter^2 / area - 1.0)")
        
        with col2:
            concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=1.0, value=0.0, help="Mean of the severity of concave portions of the contour")
            concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, max_value=1.0, value=0.0, help="Mean for the number of concave portions of the contour")
            symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, max_value=1.0, value=0.0, help="Mean of symmetry in the tumor cells")
            fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, max_value=0.1, value=0.0, help="Mean for 'coastline approximation' - 1")
        
        with col3:
            radius_se = st.number_input('Radius SE', min_value=0.0, max_value=10.0, value=0.0, help="Standard error for the mean of distances from center to points on the perimeter")
            texture_se = st.number_input('Texture SE', min_value=0.0, max_value=10.0, value=0.0, help="Standard error for the standard deviation of gray-scale values")
            perimeter_se = st.number_input('Perimeter SE', min_value=0.0, max_value=50.0, value=0.0, help="Standard error for the size of the tumor perimeter")
            area_se = st.number_input('Area SE', min_value=0.0, max_value=500.0, value=0.0, help="Standard error for the area of the tumor")
            smoothness_se = st.number_input('Smoothness SE', min_value=0.0, max_value=0.1, value=0.0, help="Standard error for local variation in radius lengths")

        # Gather the input data
        data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                          concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                          radius_se, texture_se, perimeter_se, area_se, smoothness_se]])

        submit = st.form_submit_button('🔍 Predict Cancer')
        if submit:
            with st.spinner('Analyzing...'):
                prediction = predict_cancer(data)
                if prediction == 1:
                    st.error("⚠️ **Prediction: Malignant Tumor Detected**")
                    st.markdown(
                        """
                        **Suggestion:** 
                        - Immediate consultation with an oncologist is recommended.
                        - Consider further diagnostic testing (e.g., biopsy, imaging) for confirmation.
                        - Discuss possible treatment options including surgery, chemotherapy, or radiation.
                        - Maintain a healthy lifestyle and seek support groups if needed.
                        """
                    )
                else:
                    st.success("🎉 **Prediction: Benign Tumor Detected**")
                    st.markdown(
                        """
                        **Suggestion:** 
                        - Regular follow-ups to monitor any changes in the tumor.
                        - Maintain a healthy diet and exercise routine.
                        - Keep track of any symptoms or changes and report them to a healthcare professional.
                        """
                    )
