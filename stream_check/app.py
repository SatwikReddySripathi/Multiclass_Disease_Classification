import streamlit as st
from streamlit_application.home_page import main_page

# Initialize session state
if "expanded_disease" not in st.session_state:
    st.session_state.expanded_disease = None

# Function to toggle disease expansion
def toggle_expansion(disease_id):
    if st.session_state.expanded_disease == disease_id:
        st.session_state.expanded_disease = None
    else:
        st.session_state.expanded_disease = disease_id

# Load the main page
main_page(toggle_expansion)
