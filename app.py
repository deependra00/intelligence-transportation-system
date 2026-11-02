import streamlit as st

st.set_page_config(page_title="ITS Dashboard", layout="wide")

st.title("🚗 Intelligent Transportation System (ITS) - IIoT Dashboard")

st.markdown("### Traffic Video & Analysis Stream")

# Direct video player
st.video('dataset/traffic_sample.mp4')

# Analytics cards
col1, col2, col3 = st.columns(3)
col1.metric("Vehicle Count", "18")
col2.markdown("**Density:** :orange[MEDIUM DENSITY]")
col3.metric("Accident Risk", "MODERATE RISK")