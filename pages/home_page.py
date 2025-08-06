import streamlit as st
import theme
from utility import get_weather_emoji, get_weather_sync, get_date
import asyncio


theme.custom_theme()
location = "Izmir"
date = get_date()
temperature, description = get_weather_sync(location)
emoji = get_weather_emoji(description)

st.title("Dashboard")

st.markdown(
    f"""
    <div class="header-row">
        <div class="dashboard-header">Welcome back, Efe! 👋</div>
        <div class="info-bar-inline">
            <span class="date-detail">{date}</span>
            <span class="weather-emoji">{emoji}</span>
            <span class="weather-detail">{location} {temperature}°C, {description.capitalize()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

dc_metric, at_metric = st.columns(2)
dc_metric.metric(
    label="Daily Change",
    value="$1500",
    delta="2.00%",
    border=True,
)
at_metric.metric(
    label="All-Time Change",
    value="$20k",
    delta="-8.00%",
    border=True,
)