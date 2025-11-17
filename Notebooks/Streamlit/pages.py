import streamlit as st

# Define the pages
main_page = st.Page("main_page.py", title="Project Overview", icon="🏢")
data_page = st.Page("data.py", title="Data Exploration", icon="📊")
models_page = st.Page("models.py", title="ML Models Overview", icon="🖥️")
prediction_page = st.Page("prediction.py", title="Prediction", icon="🔮")
report_page = st.Page("report.py", title="Retention Report", icon="📃")



# Set up navigation
pg = st.navigation([main_page, data_page, prediction_page, models_page,  report_page])

# Run the selected page
pg.run()