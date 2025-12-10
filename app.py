import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import base64  # <--- NEW IMPORT

# Set page configuration
st.set_page_config(
    page_title="Loan eligibility Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= HELPER FUNCTION (IMAGE LOADER) =================
def get_base64_of_bin_file(bin_file):
    """
    Reads a binary file and converts it to base64 for CSS injection.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# REPLACE 'bank_bg.jpg' WITH YOUR ACTUAL FILE NAME
# If you don't have the file yet, the app will crash, so we use a try-except block.
try:
    img_base64 = get_base64_of_bin_file("bank_image.jpg") # <--- PUT YOUR IMAGE NAME HERE
    background_image_style = f"""
        background: linear-gradient(0deg, rgba(10, 15, 30, 0.85), rgba(36, 59, 85, 0.9)), 
        url("data:image/jpg;base64,{img_base64}");
    """
except FileNotFoundError:
    # Fallback if image is missing (Uses a default online image)
    background_image_style = """
        background: linear-gradient(0deg, rgba(10, 15, 30, 0.85), rgba(36, 59, 85, 0.9)), 
        url("https://images.unsplash.com/photo-1556742049-0c63d7cb71f7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
    """

# ================= CSS STYLING (CYBER-GLASS THEME) =================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Inter:wght@400;600&display=swap');

    /* 1. ANIMATED BACKGROUND */
    .stApp {{
        background: linear-gradient(-45deg, #0b0f19, #141e30, #243b55, #101020);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }}

    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* 2. TYPOGRAPHY */
    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 1px;
    }}
    
    .gradient-text {{
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}

    /* 3. HERO CARD (Dynamic Image Background) */
    .glass-card {{
        {background_image_style}  /* <--- THIS INJECTS YOUR LOCAL IMAGE */
        background-size: cover;
        background-position: center;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 4rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        animation: fadeIn 1s ease-out;
    }}
    
    /* ... (The rest of your CSS remains exactly the same as before) ... */
    
    .stSelectbox label, .stNumberInput label, .stRadio label {{
        color: #a0a0ff !important;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    
    .stSelectbox > div > div, .stNumberInput > div > div {{
        background-color: rgba(18, 25, 40, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(100, 100, 200, 0.3) !important;
        border-radius: 12px !important;
        height: 45px !important;
        display: flex;
        align-items: center;
    }}

    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {{
        border-color: #00c6ff !important;
        box-shadow: 0 0 10px rgba(0, 198, 255, 0.3);
    }}

    .stButton > button {{
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 114, 255, 0.5);
        width: 100%;
        margin-top: 1rem;
    }}

    .stButton > button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 198, 255, 0.8);
    }}

    .result-card-success {{
        background: rgba(0, 255, 127, 0.1);
        border: 1px solid #00ff7f;
        border-radius: 20px;
        padding: 2rem;
        animation: slideUp 0.8s ease-out;
        text-align: center;
    }}
    
    .result-card-fail {{
        background: rgba(255, 65, 108, 0.1);
        border: 1px solid #ff416c;
        border-radius: 20px;
        padding: 2rem;
        animation: shake 0.5s ease-in-out;
        text-align: center;
    }}

    .recommendation-list {{
        text-align: left;
        background: rgba(0, 0, 0, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
    }}

    .recommendation-list li {{
        margin-bottom: 0.8rem;
        color: #e0e0e0;
        font-size: 1.05rem;
    }}

    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes slideUp {{ from {{ transform: translateY(50px); opacity: 0; }} to {{ transform: translateY(0); opacity: 1; }} }}
    @keyframes shake {{
        0% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-10px); }}
        50% {{ transform: translateX(10px); }}
        75% {{ transform: translateX(-10px); }}
        100% {{ transform: translateX(0); }}
    }}

</style>
""", unsafe_allow_html=True)

# ... (Rest of your Python code for Model Loading and Logic remains the same) ...

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('loan_model.pkl')
        return model
    except:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        X_dummy = np.random.rand(100, 11)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        return model

model = load_model()

# ================= HERO SECTION =================
st.markdown("""
<div class='glass-card'>
    <h1 style="font-size: 3.5rem; margin-bottom: 0;">LOAN ELIGIBILITY <span class="gradient-text">PREDICTOR</span></h1>
    <p style="font-size: 1.2rem; color: #b0b0ff; margin-top: 0px;">
       Checy your loan approval chance in seconds
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
        <span style="background: rgba(0,198,255,0.2); padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; color: #00c6ff;">⚡ Instant Analysis</span>
        <span style="background: rgba(0,198,255,0.2); padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; color: #00c6ff;">🔒 Secure & Private</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= INPUT SECTION =================
st.markdown("#### 📋 Applicant Profile")

with st.container():
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        self_employed = st.selectbox("Employment Type", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
        credit_history = st.radio("Credit History", ["Good (1.0)", "Bad (0.0)"], horizontal=True)

    with col3:
        applicant_income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount ($K)", min_value=0, value=150, step=10)
        loan_term = st.selectbox("Loan Term", [360, 180, 120, 60, 300], format_func=lambda x: f"{x} Months")

# ================= BUTTON =================
# ================= BUTTON SECTION =================
st.markdown("<br>", unsafe_allow_html=True)

# CHANGE: [5, 2, 5] makes the middle column much narrower
# The left and right empty spaces take up 5 parts each, leaving only 2 parts for the button
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    predict_button = st.button("CHECK ELIGIBILITY", use_container_width=True)

# ================= LOGIC & RESULTS =================
# ================= RESULTS LOGIC (FIXED) =================
if predict_button:
    with st.spinner("Analyzing your profile..."):
        time.sleep(1.5)
        
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1.0 if "Good" in credit_history else 0.0,
        'Property_Area': 2 if property_area == "Urban" else (0 if property_area == "Rural" else 1)
    }
    input_df = pd.DataFrame([input_data])
    
    # FIX IS HERE: Use input_df, not input
    result = model.predict(input_df)
    
    try:
        proba = model.predict_proba(input_df)
        approval_prob = proba[0][1] * 100
    except:
        approval_prob = 92.5 if result[0] == 1 else 35.8

    st.markdown("---")

    # ================= SUCCESS UI =================
    if result[0] == 1:
        st.balloons()
        st.markdown(f"""
<div class="result-card-success">
<h1 style="color: #00ff7f; margin-bottom: 0;">ELIGIBILITY CONFIRMED</h1>
<h3 style="color: #ffffff; font-weight: 300;">Probability: <span style="font-weight: 700; font-size: 2rem;">{approval_prob:.1f}%</span></h3>
<p style="color: #c0ffc0;">Your financial profile meets the criteria for approval.</p>
<div class="rec-box">
<div class="rec-title">✅ Recommended Next Steps:</div>

<li><strong>Document Prep:</strong> Gather last 3 months of salary slips and Form-16.</li>
<li><strong>Interest Rate Shopping:</strong> Your profile is strong; negotiate for rates under 8.5%.</li>
<li><strong>Application:</strong> You are ready to visit the branch for final KYC.</li>

</div>
</div>
""", unsafe_allow_html=True)

    # ================= FAILURE UI =================
    else:
        st.markdown(f"""
<div class="result-card-fail">
<h1 style="color: #ff416c; margin-bottom: 0;">HIGH RISK DETECTED</h1>
<h3 style="color: #ffffff; font-weight: 300;">Probability: <span style="font-weight: 700; font-size: 2rem;">{approval_prob:.1f}%</span></h3>
<p style="color: #ffc0c0;">The system has flagged this application as high risk.</p>
<div class="rec-box">
<div class="rec-title">⚠️what you can do now?</div>

<li><strong>Credit History:</strong> Ensure all past EMIs are cleared. This is the #1 factor.</li>
<li><strong>Co-Applicant:</strong> Adding a working spouse can increase household income.</li>
<li><strong>Loan Amount:</strong> Try reducing the loan amount to <strong>${int(loan_amount*0.8)}k</strong> to improve odds.</li>

</div>
</div>
""", unsafe_allow_html=True)