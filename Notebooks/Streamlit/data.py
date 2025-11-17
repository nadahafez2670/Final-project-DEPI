import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.title("📁 Data Exploration")
st.header("Data Sample", divider=True)

df =pd.read_csv('cleaned_dataset.csv')

df.reset_index(drop=True, inplace=True)

st.table(df.head(5))


st.header("Churn Distribution", divider=True)


counts = [int(df['churn'].sum()), int((df['churn'] == 0).sum())]
labels = ["Churned", "Not Churned"]

fig = px.pie(
    names=labels,
    values=counts,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """### **Class Imbalance Report**

    The target variable (Churn) is imbalanced:

    No (did not churn): ~44.5%

    Yes (churned): ~55.5%

    by analysing the target variable (Churn) we can see that most customers of our shop are churning!
    This imbalance indicates that the dataset is skewed towards the "yes" class, which may bias the model to predict "yes" more frequently.
    We will handle this imbalance later during the modeling stage using techniques such as resampling (e.g., SMOTE) or adjusting class weights."""
)


st.header("Gender distribution", divider=True)

gender = df["gender"].value_counts().values
label = df["gender"].value_counts().index

fig1=px.pie(
    values = gender, names = label
)

st.plotly_chart(fig1, use_container_width=True)

st.header("Churn by Gender", divider= True)


df2 = df.replace({"churn": {0: "Not Churned", 1: "Churned"}})

fig2 = px.histogram(
    df2,                 # ONLY pass df once here
    x="gender",          # x is defined here
    color="churn",       # legend groups
    barmode="group",
    color_discrete_sequence=["#94b5d2", "#b6d895"],

)
fig.update_layout(
    title="Churn by Gender",
    xaxis_title="Gender",
    yaxis_title="Count"
)
st.plotly_chart(fig2, use_container_width=True)


st.text_area("","More than half of the customers are Male, BUT The churn rate of the female customers are much higher than the male customers "
)

st.header("Churn by Support calls", divider= True)


df2 = df.replace({"churn": {0: "Not Churned", 1: "Churned"}})

fig3 = px.histogram(
    df2,                 # ONLY pass df once here
    x="support_calls",          # x is defined here
    color="churn",       # legend groups
    barmode="group",
    color_discrete_sequence=["#94b5d2", "#b6d895"],

)

st.plotly_chart(fig3, use_container_width=True)

st.text_area("","As the number of support calls increases, the churn rate rises significantly. Customers who make more than 5 calls almost always churn, which indicates potential issues with the effectiveness of customer support.")


# --- 3 Columns Layout ---
col1, col2, col3 = st.columns(3)

# =============================
# 1️⃣ Subscription Types - Pie
# =============================
with col1:
    subscription = df["subscription_type"].value_counts().reset_index()
    subscription.columns = ["subscription_type", "count"]

    fig4 = px.pie(
        subscription,
        names="subscription_type",
        values="count",
        title="Subscription Types Distribution",
        color_discrete_sequence=["#66b3ff", "#99ff99", "#ffcc99"]
    )
    st.plotly_chart(fig4, use_container_width=True)


# =====================================
# 2️⃣ Churn Rate by Subscription Type
# =====================================
with col2:
    # Calculate mean churn rate for each subscription type
    churn_by_sub = df.groupby("subscription_type")["churn"].mean().reset_index()

    fig5 = px.bar(
        churn_by_sub,
        x="subscription_type",
        y="churn",
        title="Churn Rate by Subscription Type",
        labels={"churn": "Churn Rate", "subscription_type": "Subscription Type"},
        color="subscription_type",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig2.update_yaxes(range=[0, 1])  # Because churn rate is between 0–1
    st.plotly_chart(fig5, use_container_width=True)


# =====================================
# 3️⃣ Contract Length vs Churn (Count)
# =====================================
with col3:
    fig6 = px.histogram(
        df2,
        x="contract_length",
        color="churn",
        barmode="group",
        title="Contract Length vs Churn",
        labels={"contract_length": "Contract Length", "churn": "Churn"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig6, use_container_width=True)

st.text_area("",
"""The subscription types have nearly equal proportions in the data.
The churn rate across the subscription types is also almost equal, although the Basic plan shows a slightly higher churn rate compared to the others.
Almost half of customers with annual and quarterly contracts churn, but the majorty of customers with monthly contracts churn """
)



corr = df.corr(numeric_only=True)

fig7 = px.imshow(
    corr,
    text_auto='.4f',
    color_continuous_scale="RdBu_r",   # Same idea as 'coolwarm'
    title="Full Feature Correlation Heatmap"
)

fig7.update_layout(
    width=900,
    height=900,
    xaxis_title="Features",
    yaxis_title="Features",
)

st.plotly_chart(fig7, use_container_width=True)
