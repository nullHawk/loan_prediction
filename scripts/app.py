"""
Simple Streamlit App for Loan Prediction - Fixed for PyTorch compatibility
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the project directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'src'))

# Page configuration
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="wide"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_predictor():
    """Load the predictor with caching to avoid reloading"""
    try:
        # Import only when needed
        from src.inference import LoanPredictor
        return LoanPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.title("🏦 Loan Prediction System")
    st.markdown("AI-Powered Loan Approval Decision Support")
    
    # Load model
    if st.session_state.predictor is None:
        with st.spinner("Loading model..."):
            st.session_state.predictor = load_predictor()
    
    if st.session_state.predictor is None:
        st.error("Failed to load the prediction model. Please check your setup.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose page", ["Single Prediction", "Model Info"])
    
    if page == "Single Prediction":
        single_prediction_page()
    else:
        model_info_page()

def single_prediction_page():
    st.header("📋 Single Loan Application")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Information")
        annual_inc = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
        dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=300.0, step=10.0)
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, value=5000.0, step=100.0)
    
    with col2:
        st.subheader("Credit Information")
        credit_history_length = st.number_input("Credit History Length (years)", min_value=0.0, value=10.0, step=0.5)
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        debt_to_credit_ratio = st.number_input("Debt-to-Credit Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        total_credit_lines = st.number_input("Total Credit Lines", min_value=0, value=10, step=1)
    
    # Threshold control
    st.subheader("⚙️ Prediction Settings")
    threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05, 
                         help="Higher threshold = more conservative approval")
    
    # Prediction button
    if st.button("🔮 Predict Loan Outcome", type="primary"):
        input_data = {
            'annual_inc': annual_inc,
            'dti': dti,
            'installment': installment,
            'int_rate': int_rate,
            'revol_bal': revol_bal,
            'credit_history_length': credit_history_length,
            'revol_util': revol_util,
            'debt_to_credit_ratio': debt_to_credit_ratio,
            'total_credit_lines': total_credit_lines
        }
        
        try:
            with st.spinner("Making prediction..."):
                result = st.session_state.predictor.predict_single(input_data)
            
            # Display results
            probability = result['probability_fully_paid']
            custom_prediction = 1 if probability >= threshold else 0
            
            st.subheader("📊 Prediction Results")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probability", f"{probability:.3f}")
            with col2:
                st.metric("Threshold", f"{threshold:.3f}")
            with col3:
                decision = "APPROVED" if custom_prediction == 1 else "REJECTED"
                color = "green" if custom_prediction == 1 else "red"
                st.markdown(f"<h3 style='color: {color};'>{decision}</h3>", unsafe_allow_html=True)
            
            # Explanation
            if custom_prediction == 1:
                st.success(f"✅ **LOAN APPROVED** - Probability ({probability:.3f}) ≥ Threshold ({threshold:.3f})")
            else:
                st.error(f"❌ **LOAN REJECTED** - Probability ({probability:.3f}) < Threshold ({threshold:.3f})")
            
            # Risk assessment
            if probability > 0.8:
                risk_level = "Low Risk"
                risk_color = "green"
            elif probability > 0.6:
                risk_level = "Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "High Risk"
                risk_color = "red"
            
            st.markdown(f"**Risk Level:** <span style='color: {risk_color};'>{risk_level}</span>", 
                       unsafe_allow_html=True)
            
            # Additional insights
            st.info(f"""📈 **Business Insights:**
            - Default probability: {(1-probability):.1%}
            - Confidence level: {max(probability, 1-probability):.1%}
            - Recommendation: {"Approve with standard terms" if probability > 0.8 else "Consider additional review" if probability > 0.6 else "High risk - requires careful evaluation"}
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def model_info_page():
    st.header("🤖 Model Information")
    
    st.subheader("🏗️ Model Architecture")
    st.write("""
    **Deep Artificial Neural Network (ANN)**
    - Input Layer: 9 features
    - Hidden Layer 1: 128 neurons (ReLU)
    - Hidden Layer 2: 64 neurons (ReLU)
    - Hidden Layer 3: 32 neurons (ReLU)
    - Hidden Layer 4: 16 neurons (ReLU)
    - Output Layer: 1 neuron (Sigmoid)
    - Dropout: [0.3, 0.3, 0.2, 0.1]
    """)
    
    st.subheader("📊 Input Features")
    features_df = pd.DataFrame([
        {"Feature": "annual_inc", "Description": "Annual income ($)"},
        {"Feature": "dti", "Description": "Debt-to-income ratio (%)"},
        {"Feature": "installment", "Description": "Monthly loan installment ($)"},
        {"Feature": "int_rate", "Description": "Loan interest rate (%)"},
        {"Feature": "revol_bal", "Description": "Total revolving credit balance ($)"},
        {"Feature": "credit_history_length", "Description": "Credit history length (years)"},
        {"Feature": "revol_util", "Description": "Revolving credit utilization (%)"},
        {"Feature": "debt_to_credit_ratio", "Description": "Debt to available credit ratio"},
        {"Feature": "total_credit_lines", "Description": "Total number of credit lines"}
    ])
    st.dataframe(features_df, use_container_width=True)
    
    st.subheader("📖 How to Use")
    st.write("""
    1. **Enter loan application details** in the form
    2. **Adjust the threshold slider** to control approval strictness
    3. **Click "Predict"** to get results
    4. **Interpret results:**
       - Higher threshold = more conservative (fewer approvals)
       - Lower threshold = more liberal (more approvals)
       - Probability shows model confidence in loan repayment
    """)

if __name__ == "__main__":
    main()
