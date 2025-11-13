import streamlit as st

# Define the pages
main_page = st.Page("prediction.py", title="prediction", icon="🎈")
page_1 = st.Page("report.py", title="Retention Report", icon="❄️")


# Set up navigation
pg = st.navigation([main_page, page_1])

# Run the selected page
pg.run()