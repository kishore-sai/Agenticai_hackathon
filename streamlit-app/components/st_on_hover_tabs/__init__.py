import os
import streamlit as st
import streamlit.components.v1 as components



root_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(root_dir, "frontend/build")

_on_hover_tabs = components.declare_component(
    "on_hover_tabs",
    path=build_dir
)


def on_hover_tabs(tabName, iconName, styles=None, default_choice=1, key=None):
    
    component_value = _on_hover_tabs(tabName=tabName, iconName=iconName, styles=styles, key=key, default=tabName[default_choice])
    
    return component_value

