import streamlit as st

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Page router
if st.session_state["current_page"] == "Home":
    import pages.Home_Page as home
    home.run()  # Display the Home page content
elif st.session_state["current_page"] == "Disease Prediction":
    import pages.Disease_Prediction as predictor
    predictor.run()  # Display the Disease Prediction page
