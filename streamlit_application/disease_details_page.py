import streamlit as st
from main_page import DISEASE_INFO
"""
def disease_details_page():
    disease_id = st.session_state.get("selected_disease", None)
    disease = DISEASE_INFO.get(disease_id, None)

    if not disease:
        st.error("Disease information not found.")
        if st.button("Back to Main"):
            st.session_state.page = "main"
            st.experimental_rerun()
        return

    st.title(f"Disease Information: {disease['name']}")
    st.image(disease["image"], caption=disease["name"], use_column_width=True)
    st.subheader("Description")
    st.write(disease["description"])
    st.subheader("Symptoms")
    st.write(", ".join(disease["symptoms"]))

    if st.button("Back to Main"):
        st.session_state.page = "main"
        st.experimental_rerun()
"""

def disease_details_page(navigate, disease_id):
    disease = DISEASE_INFO.get(int(disease_id), None)

    if not disease:
        st.error("Disease information not found.")
        if st.button("Back to Main Page"):
            navigate("Main Page")
        return

    st.title(f"Disease Information: {disease['name']}")
    st.image(disease["image"], caption=disease["name"], use_column_width=True)
    st.subheader("Description")
    st.write(disease["description"])
    st.subheader("Symptoms")
    st.write(", ".join(disease["symptoms"]))

    if st.button("Back to Main Page"):
        navigate("Main Page")
