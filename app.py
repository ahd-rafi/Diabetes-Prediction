import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .negative {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Diabetes Prediction App</h1>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        # You'll need to update this path to your actual model path
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model.pkl and scaler.pkl are in the same directory.")
        return None, None

# Prediction function
def predict_diabetes(features, model, scaler):
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0, 1]
    
    return prediction, probability

# Main app
def main():
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📋 Patient Information")
    
    # Input fields
    preg = st.sidebar.slider("Number of Pregnancies", 0, 17, 1)
    plas = st.sidebar.slider("Plasma Glucose Concentration", 0, 200, 120)
    pres = st.sidebar.slider("Diastolic Blood Pressure (mm Hg)", 0, 122, 70)
    skin = st.sidebar.slider("Triceps Skin Fold Thickness (mm)", 0, 99, 20)
    insu = st.sidebar.slider("2-Hour Serum Insulin (mu U/ml)", 0, 846, 79)
    mass = st.sidebar.slider("Body Mass Index", 0.0, 67.1, 25.0)
    pedi = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5)
    age = st.sidebar.slider("Age", 21, 81, 30)
    
    # Create feature array
    features = [preg, plas, pres, skin, insu, mass, pedi, age]
    
    # Prediction button
    if st.sidebar.button("🔍 Predict Diabetes", type="primary"):
        prediction, probability = predict_diabetes(features, model, scaler)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box positive">
                    <h3>⚠️ High Risk of Diabetes</h3>
                    <p><strong>Probability: {probability:.2%}</strong></p>
                    <p>Please consult with a healthcare professional for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box negative">
                    <h3>✅ Low Risk of Diabetes</h3>
                    <p><strong>Probability: {probability:.2%}</strong></p>
                    <p>Maintain a healthy lifestyle to keep your risk low.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Risk Visualization")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance explanation
    st.markdown("---")
    st.subheader("📈 Understanding the Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Health Indicators:**
        - **Pregnancies**: Number of times pregnant
        - **Glucose**: Plasma glucose concentration (mg/dL)
        - **Blood Pressure**: Diastolic blood pressure (mm Hg)
        - **Skin Thickness**: Triceps skin fold thickness (mm)
        """)
    
    with col2:
        st.markdown("""
        **Additional Factors:**
        - **Insulin**: 2-Hour serum insulin (mu U/ml)
        - **BMI**: Body mass index (weight in kg/(height in m)^2)
        - **Pedigree**: Diabetes pedigree function
        - **Age**: Age in years
        """)
    
    # Sample data visualization
    st.markdown("---")
    st.subheader("📊 Sample Data Distribution")
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'Pedigree', 'Age'],
        'Your_Values': features,
        'Average_Values': [3.8, 120.9, 69.1, 20.5, 79.8, 31.9, 0.47, 33.2]
    })
    
    fig = px.bar(sample_data, x='Feature', y=['Your_Values', 'Average_Values'], 
                 title="Your Values vs Average Values",
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Important Disclaimer:**
    This app is for educational purposes only and should not be used as a substitute for professional medical advice. 
    Always consult with a qualified healthcare provider for medical diagnosis and treatment.
    """)

if __name__ == "__main__":
    main()
