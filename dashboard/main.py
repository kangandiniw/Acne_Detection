import streamlit as st
from streamlit_option_menu import option_menu
import home
import det_acne
import about
import contact

# Load custom CSS
def load_css():
    with open("dashboard/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="SkinRenew",
    page_icon="🌻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load the CSS to apply styling
load_css()

# Create a navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "DetAcne", "About", "Contact"],
    icons=["house", "camera", "file", "phone"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Display content based on selected menu
if selected == "Home":
    home.display()  # Call the display function from home.py
elif selected == "DetAcne":
    det_acne.display()  # Call the display function from det_acne.py
elif selected == "About":
    about.display()  # Call the display function from about.py
elif selected == "Contact":
    contact.display()  # Call the display function from contact.py
