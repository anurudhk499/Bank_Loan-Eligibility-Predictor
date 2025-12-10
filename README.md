## Loan Eligibility Prediction System<br>
A Machine Learning–Driven Decision Support Web Application

### Live Demo
https://bankloan-eligibility-predictor-mfsawstzsp8vggoj8mgpob.streamlit.app/

### Project Description
The Loan Eligibility Prediction System is a full-stack machine learning web application designed to assist financial institutions and users in assessing loan approval likelihood based on applicant demographic, financial, and credit-related attributes.
The application leverages a supervised classification model to predict whether a loan applicant is eligible for approval and provides a probability score representing approval confidence. The system is deployed as an interactive web interface using Streamlit, combining machine learning inference with a modern, responsive user interface.
This project demonstrates the end-to-end ML lifecycle, including data preprocessing, model inference, UI integration, error handling, and user-centric result interpretation.

### Problem Statement
- Loan approval is traditionally a manual, time-consuming, and rule-based process that may:
- Delay decision-making
- Introduce bias
- Lack transparency for applicants

#### This project aims to:
- Automate initial eligibility screening
- Standardize decision logic
- Provide fast, explainable predictions
- Reduce human effort during pre-approval stages

### Technology Stack
#### Programming Language
Python 3.9+
#### Frontend
Streamlit<br>
Interactive forms<br>
Responsive layout with columns<br>
Client-side state handling<br>
#### Backend & Data Processing
Pandas – feature vector creation<br>
NumPy – numerical operations<br>
Joblib – model serialization and loading<br>
#### Machine Learning
Scikit-learn<br>
Random Forest Classifier<br>
Probability-based prediction (predict_proba)<br>
#### UI & Styling
Custom CSS injection<br>
Glassmorphism styling<br>
Gradient animations<br>
Conditional result rendering (success/failure)<br>
#### Machine Learning Pipeline
##### Model Type
Random Forest Classifier<br>
Handles non-linear decision boundaries<br>
Robust to feature scaling issues<br>
Suitable for structured tabular data<br>


### Project Structure
```text
loan-eligibility-predictor/
│
├── app.py                 # Main Streamlit application
├──main.ipnb               # Source code containing training models
├── loan_model.pkl         # Trained ML model
├── bank_image.jpg         # Background image
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
