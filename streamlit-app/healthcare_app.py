import streamlit as st
from streamlit.components.v1 import html
from streamlit_extras.switch_page_button import switch_page


# ---------- GLOBAL STYLES ---------- #
st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")
switch_page("overview")