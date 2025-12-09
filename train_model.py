import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime

def preprocess_data(df):
    """Data preprocessing function"""
    df = df.copy()
    
    # Drop ID column
    df = df.drop('id', axis=1)
    
    # Convert dates to datetime
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')
    
    # Calculate treatment duration in days
    df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
    
    # Drop date columns after feature engineering
    df = df.drop(['diagnosis_date', 'end_treatment_date'], axis=1)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['gender', 'country', 'cancer_stage', 'family_history', 
                       'smoking_status', 'treatment_type']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Convert binary columns (0/1)
    binary_cols = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    
    return df, label_encoders

def train_and_save_model():
    """Train model and save artifacts"""
    print("Loading dataset...")
    df = pd.read_csv('dataset_med.csv')
    
    print("Preprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop('survived', axis=1)
    y = df_processed['survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    joblib.dump(model, 'lung_cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
    print("Model training completed successfully!")
    print(f"Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    train_and_save_model()
