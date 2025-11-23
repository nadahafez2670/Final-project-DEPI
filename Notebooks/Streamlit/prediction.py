import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Prediction & Retention")

# --- CONFIGURE GEMINI API ---
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state["GOOGLE_API_KEY"] = "AIzaSyDij_OLwuQTuG5ePUct0gqTMPmTCV9leqc"  
genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])

# Load the model
# NOTE: Ensure the path is correct for your local machine or deployment
try:
    model = joblib.load('xgb_best_model.joblib') # Adjusted path for relative reference, change back if needed
except:
    st.error("Model file not found. Please check the path.")

# ==========================================
# 2. INPUT FORM (SIDEBAR)
# ==========================================
feature_names=['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
       'total_spend', 'last_interaction', 'gender_Female', 'gender_Male',
       'subscription_type_Basic', 'subscription_type_Premium',
       'subscription_type_Standard', 'contract_length_Annual',
       'contract_length_Monthly', 'contract_length_Quarterly']
numerical_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

default_values = [40,31,15,4,13,620,15,True,False,True,False,False,True,False,False]

user_inputs = {}
st.sidebar.header("Customer Profile Input")
for i, feature in enumerate(feature_names):
    if feature in numerical_features:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1 if isinstance(default_values[i], int) else 0.01
        )
    elif isinstance(default_values[i], bool):
        user_inputs[feature] = st.sidebar.checkbox(feature, value=default_values[i])
    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1
        )

input_data = pd.DataFrame([user_inputs])

# ==========================================
# 3. PAGE LAYOUT
# ==========================================
st.title("🔮 Customer Churn Prediction & Retention AI")

# Create two columns for the top section
left_col, right_col = st.columns([1, 1])

# --- FEATURE IMPORTANCE (LEFT) ---
with left_col:
    st.subheader("Feature Importance")
    try:
        feature_importance_df = pd.read_excel("xgb_importances.xlsx", usecols=["feature", "Score"])
        fig = px.bar(
            feature_importance_df.head(10).sort_values(by="Score", ascending=True),
            x="Score",
            y="feature",
            orientation="h",
            title="Top 10 Influential Factors",
            color_discrete_sequence=["#1E90FF"]
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance file (xlsx) not found. Skipping plot.")

# --- PREDICTION LOGIC (RIGHT) ---
with right_col:
    st.subheader("Churn Risk Analysis")
    
    # Initialize session state for prediction result if it doesn't exist
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.is_churn = False

    if st.button("Run Prediction Model", type="primary"):
        probabilities = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]
        
        # Save to session state so it persists
        st.session_state.prediction_made = True
        st.session_state.churn_prob = probabilities[1]
        st.session_state.retain_prob = probabilities[0]
        st.session_state.is_churn = (prediction == 1)
        
        # Reset chat history if a new prediction is made
        st.session_state.messages = []

    # Display Results if prediction exists
    if st.session_state.prediction_made:
        if st.session_state.is_churn:
            st.error(f"### ⚠️ Status: HIGH CHURN RISK")
            st.write(f"**Probability of Churn:** {st.session_state.churn_prob:.2%}")
            st.write("This customer is likely to leave. **Retention Protocol Activated.**")
        else:
            st.success(f"### ✅ Status: LOYAL CUSTOMER")
            st.write(f"**Probability of Retention:** {st.session_state.retain_prob:.2%}")

st.markdown("---")

# ==========================================
# 4. AI RETENTION AGENT (THE GEMINI INTEGRATION)
# ==========================================

# Only show the chatbot if the customer is CHURNING (Risk > 50%)
if st.session_state.get("is_churn", False):
    
    st.subheader("🤖 AI Retention Specialist (Live Demo)")
    st.caption("The system has flagged this customer. The AI is now attempting to negotiate via chat.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- DEFINE THE SYSTEM PROMPT ---
    system_instruction = f"""
    You are 'Sarah', a Senior Customer Retention Specialist. 
    You are speaking to a customer who is highly likely to cancel their service (Churn Risk: {st.session_state.churn_prob:.2%}).
    
    Your Goal: De-escalate the situation, understand their frustration, and convince them to stay.
    
    Your Personality: Professional, Empathetic, Apologetic, but Solution-Oriented.
    
    You have the following "Retention Toolkit" (Offers) you can use ONE BY ONE if needed:
    1. For Technical Issues (High Support Calls): Offer 'Priority VIP Support' access.
    2. For Pricing Issues (Monthly Contract): Offer a 20% discount if they switch to Annual.
    3. For Payment Issues: Offer a 14-day payment extension.
    4. For General Unhappiness: Offer 1 free month of service.
    
    Context about the customer:
    - Support Calls: {user_inputs.get('support_calls', 'Unknown')} (If > 5, acknowledge their frustration with support).
    - Contract: {'Monthly' if user_inputs.get('contract_length_Monthly') else 'Long-term'}.
    - Total Spend: ${user_inputs.get('total_spend', 0)}.
    
    Start the conversation by acknowledging their potential dissatisfaction gently. Keep responses short (under 50 words) and conversational.
    """

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # If history is empty, AI sends the first message
    if not st.session_state.messages:
        initial_msg = "Hello. I noticed you've had some recent difficulties with your account. I'm Sarah, a Senior Specialist here. How can I help resolve things for you today?"
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        with st.chat_message("assistant"):
            st.markdown(initial_msg)

    # Chat Input
    if prompt := st.chat_input("Reply as the Customer..."):
        # 1. Display User Message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Generate Gemini Response
        try:
            # Construct the full history for the API
            chat_history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state.messages]
            
            # Initialize model with system instruction (using gemini-2.0-flash for speed)
            model_ai = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_instruction)
            
            # Generate response
            response = model_ai.generate_content(chat_history)
            ai_reply = response.text
            
            # 3. Display AI Message
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            st.error(f"Error connecting to Gemini: {e}")

elif st.session_state.get("prediction_made", False) and not st.session_state.is_churn:
    st.info("👋 This customer is Low Risk. No retention intervention is needed.")