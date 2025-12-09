import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Survival Predictor",
    page_icon="🫁",
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
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .survive {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .not-survive {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_models():
    model = joblib.load('lung_cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, label_encoders, feature_names

# Main app
def main():
    st.markdown('<h1 class="main-header">🫁 Lung Cancer Survival Prediction System</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    try:
        model, scaler, label_encoders, feature_names = load_models()
    except Exception as e:
        st.error("⚠️ Model files nahi mile! Pehle `train_model.py` run karo.")
        st.stop()
    
    # Sidebar inputs
    st.sidebar.markdown('<h2 class="sub-header">📋 Patient Information</h2>', 
                       unsafe_allow_html=True)
    
    # Personal Info
    st.sidebar.subheader("👤 Personal Details")
    age = st.sidebar.slider("Age", 18, 90, 50)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    # Medical History
    st.sidebar.subheader("🏥 Medical History")
    cancer_stage = st.sidebar.selectbox(
        "Cancer Stage", 
        ["Stage I", "Stage II", "Stage III", "Stage IV"]
    )
    family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", 
        ["Current Smoker", "Former Smoker", "Never Smoked", "Passive Smoker"]
    )
    
    # Health Metrics
    st.sidebar.subheader("📊 Health Metrics")
    bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
    cholesterol = st.sidebar.number_input("Cholesterol Level", 100, 400, 200)
    
    # Conditions
    st.sidebar.subheader("🩺 Medical Conditions")
    hypertension = st.sidebar.checkbox("Hypertension")
    asthma = st.sidebar.checkbox("Asthma")
    cirrhosis = st.sidebar.checkbox("Cirrhosis")
    other_cancer = st.sidebar.checkbox("Other Cancer History")
    
    # Treatment
    st.sidebar.subheader("💊 Treatment Details")
    treatment_type = st.sidebar.selectbox(
        "Treatment Type", 
        ["Surgery", "Chemotherapy", "Radiation", "Combined"]
    )
    treatment_duration = st.sidebar.number_input(
        "Treatment Duration (days)", 
        0, 1000, 180
    )
    
    # Country
    country = st.sidebar.selectbox(
        "Country", 
        ["Sweden", "Netherlands", "Hungary", "Belgium", "Luxembourg", 
         "Italy", "Croatia", "Denmark", "Germany", "Malta", "Poland", 
         "Ireland", "Romania", "Spain", "Greece", "Estonia", "Cyprus", "France"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🔍 Patient Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Create input dataframe
        input_data = {
            'age': age,
            'gender': label_encoders['gender'].transform([gender])[0],
            'country': label_encoders['country'].transform([country])[0],
            'cancer_stage': label_encoders['cancer_stage'].transform([cancer_stage])[0],
            'family_history': label_encoders['family_history'].transform([family_history])[0],
            'smoking_status': label_encoders['smoking_status'].transform([smoking_status])[0],
            'bmi': bmi,
            'cholesterol_level': cholesterol,
            'hypertension': int(hypertension),
            'asthma': int(asthma),
            'cirrhosis': int(cirrhosis),
            'other_cancer': int(other_cancer),
            'treatment_type': label_encoders['treatment_type'].transform([treatment_type])[0],
            'treatment_duration': treatment_duration
        }
        
        # Display input summary
        st.write("### Patient Summary")
        summary_df = pd.DataFrame({
            'Feature': ['Age', 'Gender', 'Cancer Stage', 'Smoking Status', 
                       'BMI', 'Treatment Type'],
            'Value': [age, gender, cancer_stage, smoking_status, 
                     f"{bmi:.1f}", treatment_type]
        })
        st.table(summary_df)
        
        # Prediction button
        if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_names]  # Ensure correct column order
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display result
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-box survive">✅ Patient is likely to SURVIVE</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-box not-survive">⚠️ Patient is at HIGH RISK</div>',
                    unsafe_allow_html=True
                )
            
            # Probability gauge
            st.write("### Survival Probability")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba[1] * 100,
                title={'text': "Survival Chance (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Survival Probability", f"{prediction_proba[1]*100:.2f}%")
            with col_b:
                st.metric("Risk Probability", f"{prediction_proba[0]*100:.2f}%")
    
    with col2:
        st.markdown('<h2 class="sub-header">📈 Risk Factors</h2>', 
                   unsafe_allow_html=True)
        
        # Risk factor analysis
        risk_factors = []
        risk_scores = []
        
        if cancer_stage in ["Stage III", "Stage IV"]:
            risk_factors.append("Advanced Stage")
            risk_scores.append(80)
        
        if smoking_status in ["Current Smoker", "Former Smoker"]:
            risk_factors.append("Smoking History")
            risk_scores.append(70)
        
        if age > 65:
            risk_factors.append("Advanced Age")
            risk_scores.append(60)
        
        if other_cancer:
            risk_factors.append("Other Cancer")
            risk_scores.append(65)
        
        if hypertension or cirrhosis:
            risk_factors.append("Comorbidities")
            risk_scores.append(50)
        
        if bmi < 18.5 or bmi > 30:
            risk_factors.append("BMI Issues")
            risk_scores.append(40)
        
        if risk_factors:
            risk_df = pd.DataFrame({
                'Risk Factor': risk_factors,
                'Impact Score': risk_scores
            })
            
            fig = px.bar(
                risk_df, 
                y='Risk Factor', 
                x='Impact Score',
                orientation='h',
                color='Impact Score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No major risk factors detected!")
        
        # Feature importance (top features)
        st.write("### 🔑 Key Factors")
        importance_df = pd.DataFrame({
            'Feature': ['Cancer Stage', 'Age', 'Treatment Duration', 
                       'BMI', 'Smoking Status'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        
        fig2 = px.pie(
            importance_df, 
            values='Importance', 
            names='Feature',
            hole=0.4
        )
        fig2.update_layout(
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>⚕️ This is a predictive model for educational purposes only. 
        Always consult healthcare professionals for medical decisions.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
