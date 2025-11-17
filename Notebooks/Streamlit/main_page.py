import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis Project",
    page_icon="📊",
    layout="wide"
)


st.title("Customer Churn Analysis & Retention Strategy")


st.markdown("""
This project helps students analyze customer behavior and the factors contributing to
customer attrition, and use the derived insights to inform a data-backed customer
retention strategy. The core deliverable is an end-to-end machine learning solution
designed to predict customer churn proactively. 

This initiative is critical as **customer acquisition costs are significantly higher than retention costs**, 
making the ability to accurately forecast and mitigate churn a key competitive advantage 
for any subscription-based or service-oriented business. 

The project demonstrates the application of advanced data science techniques—from exploratory data analysis 
and rigorous preprocessing to sophisticated model development and practical deployment—to solve a high-value, real-world business problem.
""")

st.subheader("Key Highlights")
st.markdown("""
- **Analyze Customer Behavior**: Understand patterns leading to churn.
- **Predictive Modeling**: Build models to forecast churn proactively.
- **Retention Strategy**: Translate insights into actionable business decisions.
- **End-to-End Workflow**: From data exploration to deployment.
""")


