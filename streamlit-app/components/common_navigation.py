import streamlit as st
from components.st_on_hover_tabs import on_hover_tabs
from streamlit_javascript import st_javascript
import os
import uuid
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd


class Dependencies:

    @staticmethod
    def common_tabs():
        with st.sidebar:
            selected_tab = on_hover_tabs(tabName=['','Overview', 'ChatBot', 'Dashboards'],
                                         iconName=['','home', 'link', 'folder'], default_choice=0)


            url  = st_javascript("await fetch('').then(r => window.parent.location.href)")

            if url and url.rsplit('/', 1)[1] == selected_tab.lower():
                pass
            elif selected_tab == 'Overview':
                st.switch_page("pages/overview.py")
            elif selected_tab == 'ChatBot':
                st.switch_page("pages/chatbot.py")
            elif selected_tab == 'Dashboards':
                st.switch_page("pages/dashboards.py")
            
            else:
                pass

    @staticmethod
    def load_css():
        st.markdown('<style>' + open(f'{os.getcwd()}/assets/styles/style.css').read() + '</style>', unsafe_allow_html=True)

    @staticmethod
    def nav_bar():
        st.markdown("""
        <nav class="navbar">
          <span class="nav-colored-text">MediQuery</span>
          <span class="nav-colored-text2">Query. Clarity. Care.</span>
        </nav>
        """, unsafe_allow_html=True)


    