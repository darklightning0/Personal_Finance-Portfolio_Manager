
import os   
import plot

#Finance
import yfinance as yf

#Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

#Data and Calculations
import numpy as np
import ta
import pandas as pd
import datetime as dt


av_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')


home_page = st.Page("pages/home_page.py", title="Dashboard", icon="🏠")
watchlist_page = st.Page("pages/watchlist_page.py", title="Watchlist", icon="👀")


pg = st.navigation([home_page, watchlist_page])

pg.run()

#plot.graph("AMZN", "5y", "1d", ma_period=26, ma_types=('ST_SMA', 'LT_SMA'))
#plot.candlestick("AMZN", "1y", "1d")
#plot.correlation_heatmap(['MSFT', 'AAPL', 'GOOG', 'AMZN', 'TSLA', 'ASELS.IS'])