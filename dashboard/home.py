import streamlit as st

def display():
    st.title("Welcome to SkinRenew!")
    st.subheader("Your tagline goes here 🚀")
    st.write("This is the main landing page.")
    st.image("assets/logo.png", caption="Amazing SkinRenew App!")
    st.markdown("---")
    st.markdown("<p style='text-align:center;'>© 2024 SkinRenew. All Rights Reserved.</p>", unsafe_allow_html=True)
