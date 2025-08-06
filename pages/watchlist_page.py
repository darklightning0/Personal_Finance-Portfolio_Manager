import streamlit as st
import theme
import datetime as dt
import plot
from utility import (
    get_symbol_list, get_stock_profile, get_stock_quote, get_stock_metrics, get_stock_lookup,
    search_component, info_card, render_graph, indicator_system, is_market_open
)
from streamlit_autorefresh import st_autorefresh

theme.custom_theme()

st.title("Watchlist")

if "indicators" not in st.session_state:
    st.session_state.indicators = []

symbols = get_symbol_list()
selected_symbol = search_component(symbols)

if selected_symbol:
    info_card(selected_symbol, get_stock_profile, get_stock_quote, get_stock_metrics, get_stock_lookup)

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Chart Settings")
            VALID_PERIODS = {
                "1d": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
                "1h": ["1d", "3d", "1wk", "1mo", "3mo", "6mo", "1y"],
                "5m": ["1d", "3d", "1wk", "1mo"]
            }
            interval = st.selectbox(
                "Interval", options=list(VALID_PERIODS.keys()), index=0, key="interval_select"
            )
            period_options = VALID_PERIODS[interval]
            period = st.selectbox(
                "Period", options=period_options, index=3 if interval == "1d" else len(period_options) - 1, key="period_select"
            )
            chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"], key="chart_type_select")

        with col2:
            st.markdown("##### Technical Indicators")
            indicator_system()
        
        st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        show_graph = c3.checkbox("Show Graph", value=True)
        realtime = c4.checkbox("Real-Time", value=False)


    if realtime and is_market_open() == "True":
        st_autorefresh(interval=15000, limit=1000, key="rt_refresh")

  
    
    render_graph(
        symbol=selected_symbol,
        show=show_graph,
        interval=interval,
        period=period,
        chart_type=chart_type,
        realtime=realtime
    )


