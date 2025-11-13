import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

#XGB Model is chosed to predict the churn as it is the best model based on the tuning results
model = joblib.load('xgb_best_model.joblib')

feature_names=['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
       'total_spend', 'last_interaction', 'gender_Female', 'gender_Male',
       'subscription_type_Basic', 'subscription_type_Premium',
       'subscription_type_Standard', 'contract_length_Annual',
       'contract_length_Monthly', 'contract_length_Quarterly']
numerical_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']


default_values = [40,31,15,4,13,620,15,True,False,True,False,False,True,False,False]

user_inputs = {}
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

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])


st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
with left_col:
    st.header("Feature Importance")
    # Load feature importance data from the Excel file
    feature_importance_df = pd.read_excel("xgb_importances.xlsx", usecols=["feature", "Score"])
    # Plot the feature importance bar chart
    fig = px.bar(
        feature_importance_df.head(10).sort_values(by="Score", ascending=True),
        x="Score",
        y="feature",
        orientation="h",
        title="Feature Importance",
        labels={"Feature Importance Score": "Importance", "Feature": "Features"},
        width=400,  # Set custom width
        height=500  # Set custom height
    )
    st.plotly_chart(fig)

# Right Page: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        # Get the predicted probabilities and label
        probabilities = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]
        # Map prediction to label
        prediction_label = "Churned" if prediction == 1 else "Retain"

        # Display results
        st.subheader(f"Predicted Value: {prediction_label}")
        st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
        st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
        # Display a clear output for the prediction
        st.markdown(f"### Output: **{prediction_label}**")
