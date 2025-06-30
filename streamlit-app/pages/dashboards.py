import os
import pandas as pd
import streamlit as st
from PIL import Image
import os
from components.common_navigation import Dependencies
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import json

im = Image.open(f"{os.getcwd()}/assets/tiger_related_icons/tiger_favicon.png")
st.set_page_config(page_title="Healthcare", layout="wide", page_icon=im)
Dependencies.common_tabs()
Dependencies.load_css()
Dependencies.nav_bar()


# --- Function to load metadata and extract accessible tables ---
def get_accessible_tables(
    role: str, metadata_path: str = "./Metadata/table_level_metadata.json"
):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    accessible_tables = []
    for table in metadata:
        if role in table.get("roles_allowed", []):
            accessible_tables.append(table)
    return accessible_tables


# --- Overview Page ---
def render_overview_page():
    st.title("📊 Data Overview Dashboard")

    # Get current role from session
    role = st.session_state.get("role", "unknown").replace("_", " ").title()

    st.markdown(f"### 👋 Welcome, **{role}**")
    st.markdown(f"#### 🏥 Access Level: `{st.session_state.get('role')}`")

    # Fetch and display tables
    tables = get_accessible_tables(st.session_state.get("role"))

    st.markdown(f"### 🗂️ Accessible Tables ({len(tables)})")

    if not tables:
        st.warning("No tables are accessible with your role.")
        return

    for table in tables:
        with st.expander(f"🧾 {table['table_name']} - {table['description']}"):
            columns = table.get("columns", [])
            if not columns:
                st.info("No column details available.")
            else:
                for col in columns:
                    col_required = "✅" if col.get("required", False) else "❌"
                    st.markdown(
                        f"- **{col['column_name']}** ({col['data_type']}): {col['description']} {col_required}"
                    )

    st.markdown("---")
    if st.button("🔙 Logout"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.experimental_rerun()
