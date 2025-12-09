# 🫁 Lung Cancer Survival Prediction System

## 📌 Project Overview
This project is a Machine Learning application designed to predict the survival likelihood of lung cancer patients based on their medical history, diagnosis details, and treatment information. The system utilizes a **Random Forest Classifier** to analyze patient data and provides an interactive web interface built with **Streamlit** for real-time predictions and risk analysis.

## 📊 Dataset Information
The dataset contains comprehensive patient information regarding lung cancer mortality.

| Column | Description |
|--------|-------------|
| **age** | Patient's age at diagnosis |
| **gender** | Patient's gender (Male/Female) |
| **country** | Country of residence |
| **cancer_stage** | Stage of cancer (I, II, III, IV) |
| **family_history** | History of cancer in family (Yes/No) |
| **smoking_status** | Smoker status (Current, Former, Never, Passive) |
| **bmi** | Body Mass Index |
| **cholesterol_level** | Cholesterol level |
| **hypertension** | Presence of high blood pressure |
| **asthma** | Presence of asthma |
| **cirrhosis** | Presence of liver cirrhosis |
| **other_cancer** | History of other cancer types |
| **treatment_type** | Type of treatment (Surgery, Chemo, Radiation, Combined) |
| **survived** | Target variable (0: No, 1: Yes) |
| **dates** | Diagnosis and treatment end dates (used to calculate duration) |

## ⚙️ Methodology

### 1. Data Preprocessing
- **Feature Engineering**: Calculated `treatment_duration` (in days) from `diagnosis_date` and `end_treatment_date`.
- **Cleaning**: Dropped `id` and original date columns.
- **Encoding**: 
  - `LabelEncoder` for categorical variables (Gender, Country, Stage, etc.).
  - Binary conversion (0/1) for conditions (Hypertension, Asthma, etc.).
- **Imputation**: Missing values filled with median values.
- **Scaling**: Applied `StandardScaler` to normalize feature values.

### 2. Model Architecture
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - `n_estimators`: 200
  - `max_depth`: 15
  - `min_samples_split`: 5
- **Validation**: 80-20 Train-Test split with stratification.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Joblib

## 🚀 Installation & Setup

### 1. Clone the Repository

git clone https://github.com/aiml-developer/Lung-Cancer-Survival-Prediction-System.git
cd Lung-Cancer-Survival-Prediction-System


### 2. Create Virtual Environment

Windows
python -m venv venv
.\venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate


### 3. Install Dependencies

pip install -r requirements.txt


**requirements.txt content:**

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
joblib


## 🧠 Model Training
To train the model and generate the necessary `.pkl` files (model, scaler, encoders), run the training script:

python train_model.py

*Output: This will save `lung_cancer_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, and `feature_names.pkl` in your project directory.*

## 💻 Usage
Run the Streamlit application:

streamlit run app.py

The app will open in your default browser at `http://localhost:8501`.

## 📱 Demo Features
- **Sidebar Controls**: Input patient details including Age, Medical History, and Treatment info.
- **Real-time Prediction**: Instantly predicts "Likely to Survive" or "High Risk".
- **Survival Probability**: Gauge chart showing the confidence percentage.
- **Risk Analysis**: 
  - **Risk Factor Chart**: Visualizes high-impact factors like Stage IV cancer or Smoking history.
  - **Key Factors**: Pie chart highlighting which features contributed most to the prediction.

## 📈 Results
The Random Forest model is evaluated using Accuracy, Precision, Recall, and F1-Score.
- **Target Accuracy**: ~90%+ (Dependent on dataset quality)
- **Key Predictors**: Cancer Stage, Treatment Duration, and Age were found to be the most significant factors affecting survival rates.

## 🏁 Conclusion
This tool serves as an educational aid to demonstrate how Machine Learning can assist in medical prognosis. It highlights the importance of early diagnosis and lifestyle factors (like smoking and BMI) in cancer survival rates.

---
*Disclaimer: This is a student project for educational purposes. Do not use for actual medical diagnosis.*
