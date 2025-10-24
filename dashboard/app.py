#dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

API_URL = "http://127.0.0.1:8000/get_data"

st.set_page_config(page_title="Smart Borewell Monitor", layout="wide")
st.title("💧 Smart Borewell Water Quality Dashboard")

# Auto-refresh every n seconds
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 120, 5)
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# Fetch data
def fetch_data():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            st.warning("Failed to fetch data from backend.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df = fetch_data()
if not df.empty:
    df["Status"] = df["potable"].apply(lambda x: "Safe ✅" if x == 1 else "Unsafe ❌")

    st.subheader("🌍 Latest Borewell Readings")
    st.dataframe(df.sort_values(by="timestamp", ascending=False))

    st.subheader("📊 Potability by Region")
    region_counts = df.groupby(["region", "Status"]).size().reset_index(name="Count")
    fig_region = px.bar(
        region_counts,
        x="region",
        y="Count",
        color="Status",
        barmode="group",
        title="Region-wise Safe/Unsafe Count"
    )
    st.plotly_chart(fig_region, use_container_width=True, key=f"region_chart_{pd.Timestamp.now().timestamp()}")

    st.subheader("📈 Trends Over Time")
    numeric_cols = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    for col in numeric_cols:
        fig = px.line(df, x="timestamp", y=col, color="region", title=f"{col} Trend")
        st.plotly_chart(fig, use_container_width=True, key=f"{col}_chart_{pd.Timestamp.now().timestamp()}")
