
import os   
import streamlit as st




AV_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FINNHUB_API_KEY = st.secrets.get("finnhub_api_key")

home_page = st.Page("pages/home_page.py", title="Dashboard", icon="🏠")
watchlist_page = st.Page("pages/watchlist_page.py", title="Watchlist", icon="👀")
portfolio_page = st.Page("pages/portfolio_page.py", title="Portfolio", icon="💼")



pg = st.navigation([home_page, watchlist_page, portfolio_page])

pg.run()
