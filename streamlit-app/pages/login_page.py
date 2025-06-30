import streamlit as st
from streamlit.components.v1 import html
from streamlit_extras.switch_page_button import switch_page


# ---------- GLOBAL STYLES ---------- #

st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")
users = {
    "alice": {"password": "alice123", "role": "non_clinical_staff"},
    "bob": {"password": "bob456", "role": "manager"},
    "admin": {"password": "admin789", "role": "administrative_staff"}
}



# --- INITIAL SESSION STATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_error" not in st.session_state:
    st.session_state.login_error = ""

# --- PAGE CONFIG ---


# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
        .login-box {
            max-width: 400px;
            margin: auto;
            padding: 2rem;
            border-radius: 12px;
            background-color: #f9f9f9;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', sans-serif;
        }
        .login-title {
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .footer-note {
            text-align: center;
            font-size: 0.85rem;
            margin-top: 1rem;
            color: #777;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOGIN FORM ---
if not st.session_state.authenticated:
    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔐 Healthcare KPI Assistant Login</div>', unsafe_allow_html=True)

        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login", use_container_width=True):
            if username in users and users[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.role = users[username]["role"]
                st.session_state.login_error = ""
                st.success(f"✅ Welcome {username} ({st.session_state.role.replace('_', ' ').title()})")
                switch_page("overview")
                
            else:
                st.session_state.login_error = "❌ Invalid username or password."

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        st.markdown('<div class="footer-note">Contact admin if you forgot your credentials.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
