import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load model and scaler
model = xgb.XGBClassifier()
model.load_model('model.json')
scaler = joblib.load('scaler.pkl')

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Streamlit app
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on patient data using an XGBoost model.
Use the sliders below to input patient details and get a prediction.
""")

# Sidebar for user input
st.sidebar.header("Input Patient Data")

def user_input_features():
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    plas = st.sidebar.slider('Glucose', 0, 199, 117)
    pres = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insu = st.sidebar.slider('Insulin', 0, 846, 30)
    mass = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    pedi = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    data = {
        'preg': preg,
        'plas': plas,
        'pres': pres,
        'skin': skin,
        'insu': insu,
        'mass': mass,
        'pedi': pedi,
        'age': age
    }
    features = pd.DataFrame(data, index=[0])[feature_names]  # Ensure correct column order
    return features

# Get user input
input_df = user_input_features()

# Scale input
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)[0]

# Display results
st.subheader("Prediction")
result = 'High Risk' if prediction[0] == 1 else 'Low Risk'
st.write(f"**Prediction**: {result}")
st.write(f"**Probability of Diabetes**: {prediction_proba[1]:.2%}")

# Plot probability gauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction_proba[1] * 100,
    title={'text': "Diabetes Risk Probability (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [50, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "red"}
        ]
    }
))
st.plotly_chart(fig_gauge)

# Plot feature comparison
input_values = input_df.iloc[0].values
avg_values = pd.read_csv('data/dataset_37_diabetes.csv')[feature_names].mean().values

fig_bar = px.bar(
    x=feature_names,
    y=input_values,
    labels={'x': 'Features', 'y': 'Values'},
    title="Input Features vs. Dataset Average"
)
fig_bar.add_scatter(x=feature_names, y=avg_values, mode='markers', name='Dataset Average')
st.plotly_chart(fig_bar)

# Disclaimer
st.markdown("""
**Disclaimer**: This app is for educational purposes only and should not be used for medical diagnosis. 
Please consult a healthcare professional for medical advice.
""")
