
# --- Standard Library Imports ---
import datetime as dt
import asyncio
import os

import plot

# --- Third-Party Imports ---
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import python_weather as pw

try:
    from zoneinfo import ZoneInfo 
except ImportError:
    from pytz import timezone as ZoneInfo 

def get_weather_emoji(description):
    """Return an emoji representing the weather description."""
    desc = description.lower()
    if "sun" in desc or "clear" in desc:
        return "☀️"
    if "cloud" in desc or "overcast" in desc:
        return "☁️"
    if "rain" in desc or "shower" in desc or "drizzle" in desc:
        return "🌧️"
    if "storm" in desc or "thunder" in desc:
        return "⛈️"
    if "snow" in desc:
        return "❄️"
    if "fog" in desc or "mist" in desc:
        return "🌫️"
    return "🌡️"


@st.cache_data(ttl=600)
def get_weather_sync(location):
    """Fetch and cache weather data for a location. Returns (temperature, description) as serializable types."""
    async def fetch():
        async with pw.Client(unit=pw.METRIC) as client:
            weather = await client.get(location)
            return float(weather.temperature), str(weather.description)
    return asyncio.run(fetch())

def get_date():
    """Return the current date as a formatted string."""
    return dt.datetime.now().strftime("%A, %d %B %Y")

FINNHUB_API_KEY = st.secrets.get("finnhub_api_key")
SYMBOL_CACHE_FILE = "symbol_cache.csv"


def round_digit(value, digit):

    if isinstance(value, (int, float)): 
        return round(value, digit)
    
    else:
        return "N/A"


def fetch_symbols():
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            st.error(f"Finnhub symbol API error: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty or not all(col in df.columns for col in ['symbol', 'displaySymbol', 'description', 'type']):
            st.error("Finnhub symbol data missing expected columns.")
            return pd.DataFrame()

        df = df[df['type'] == 'Common Stock'][['symbol', 'displaySymbol', 'description']]
        df.to_csv(SYMBOL_CACHE_FILE, index=False)

        return df

    except Exception as e:

        st.error(f"Failed to fetch symbols: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_symbol_list():
    """Return cached symbol list, fetching if not present."""
    if os.path.exists(SYMBOL_CACHE_FILE):
        try:
            return pd.read_csv(SYMBOL_CACHE_FILE)
        except Exception as e:
            st.warning(f"Could not read symbol cache: {e}. Refetching...")
    return fetch_symbols()

def get_stock_data(url):
    """Helper to fetch stock data from a URL, returns dict or None."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.warning(f"API request failed: {e}")
        return None

def get_stock_quote(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    return get_stock_data(url)

def get_stock_profile(symbol):
    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
    return get_stock_data(url)

def get_stock_lookup(symbol):
    url = f"https://finnhub.io/api/v1/search?q={symbol}&token={FINNHUB_API_KEY}"
    return get_stock_data(url)


def get_stock_metrics(symbol):
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_API_KEY}"
    data = get_stock_data(url)
    return data.get("metric", {}) if data else {}


def is_market_open(exchange='US'):

    try:
        url = f"https://finnhub.io/api/v1/stock/market-status?exchange={exchange}&token={FINNHUB_API_KEY}"
        dat = get_stock_data(url)

        if dat.get("isOpen") == False and dat.get("holiday"):
            return f"the market is closed due to {dat.get("holiday")}."

        elif dat.get("isOpen") == False and not dat.get("holiday"):
            return "the market is out of session."

        else:
            return "True"

    except Exception as e:
        st.warning(f"Could not determine market open status: {e}")
        return False

def search_component(symbols):
    """Creates a search box with autocomplete and session state for selected symbol."""
    query = st.text_input("Search for a stock...", placeholder="Type a ticker or company name", key="search_query")
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    if not query:
        st.session_state.selected_symbol = None
        return None
    exact_match_ticker = symbols[symbols['symbol'].str.lower() == query.lower()]
    exact_match_name = symbols[symbols['description'].str.lower() == query.lower()]
    if not exact_match_ticker.empty:
        st.session_state.selected_symbol = exact_match_ticker.iloc[0]['symbol']
        return st.session_state.selected_symbol
    if not exact_match_name.empty:
        st.session_state.selected_symbol = exact_match_name.iloc[0]['symbol']
        return st.session_state.selected_symbol
    matches = symbols[symbols['symbol'].str.contains(query, case=False) |
                      symbols['description'].str.contains(query, case=False)].head(5)
    if not matches.empty:
        suggestions = [f"{row['symbol']} - {row['description']}" for _, row in matches.iterrows()]
        selected = st.selectbox("Results", suggestions, index=None, placeholder="Select a stock from the results")
        if selected:
            symbol = selected.split(" - ")[0]
            st.session_state.selected_symbol = symbol
            return symbol
    st.session_state.selected_symbol = None
    return None

def get_last_price(symbol):
    """Fetches the last price of a stock symbol."""
    quote = get_stock_quote(symbol)
    if quote and 'c' in quote:
        return round_digit(quote['c'], 2)
    return "N/A"


def info_card(symbol, get_stock_profile, get_stock_quote, get_stock_metrics, get_stock_lookup):
    """Displays a detailed information card for the selected stock."""
    profile = get_stock_profile(symbol)
    quote = get_stock_quote(symbol)
    metrics = get_stock_metrics(symbol)
    lookup = get_stock_lookup(symbol)['result'][0]
    yf_data = yf.Ticker(symbol)
    if not profile or not quote:
        st.error("Could not retrieve all data for this stock.")
        return
    st.markdown(f"### {profile.get('name', symbol)} ({symbol})")
    
    with st.expander(label = "Info", expanded=True):
        cols = st.columns([2, 2, 2, 2])
        cols[0].markdown(f"**Sector:**<br><span style='font-size: 1.5em; color:#2196f3; word-wrap:break-word;'>{profile.get('finnhubIndustry', 'N/A')}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"**Stock Symbol:**<br><span style='font-size: 1.5em; color:#ffb300; word-wrap:break-word;'>{lookup.get('type', 'N/A')}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"**Exchange:**<br><span style='font-size: 1.5em; color:#ff3737; word-wrap:break-word;'>{profile.get('exchange', 'N/A')}</span>", unsafe_allow_html=True)
        cols[3].markdown(f"**Market Cap (M):**<br><span style='font-size: 1.5em; color:#00e676; word-wrap:break-word;'>${metrics.get('marketCapitalization', 'N/A')}</span>", unsafe_allow_html=True)
        cols = st.columns([2, 2, 2, 2])
        cols[0].metric(f"**Last Price:**", f"{round_digit(quote.get('c', 'N/A'), 2)}", delta=f"{round_digit(quote.get('d', 'N/A'), 2)} (%{round_digit(quote.get('dp', 'N/A'), 2)}%)", delta_color="normal")
        cols[1].metric(f"**Open Price:**", f"{quote.get('o', 'N/A')}")
        cols[2].metric(f"**Highest Price:**", f"{quote.get('h', 'N/A')}")
        cols[3].metric(f"**Lowest Price:**", f"{quote.get('l', 'N/A')}")

        with st.expander("Show All Financial Metrics"):
            with st.expander("Company Description", expanded=False):
                st.markdown(f"{yf_data.info.get("longBusinessSummary", "No description found for the selected symbol.")} ({symbol})")
            cols = st.columns([2, 2, 2, 2])
            cols[0].metric(f"**P/E Ratio:**", f"{round_digit(metrics.get('peBasicExclExtraTTM', 'N/A'), 2)}")
            cols[1].metric(f"**PEG Ratio:**", f"{round_digit(metrics.get('pegTTM', 'N/A'), 2)}")
            cols[2].metric(f"**P/B Ratio:**", f"{round_digit(metrics.get('pb', 'N/A'), 2)}")
            cols[3].metric(f"**P/S Ratio:**", f"{round_digit(metrics.get('psTTM', 'N/A'), 2)}")
            cols = st.columns([2, 2, 2, 2])
            cols[0].metric(f"**EPS (TTM):**", f"{round_digit(metrics.get('epsTTM', 'N/A'), 2)}")
            cols[1].metric(f"**Gross Margin (TTM):**", f"{round_digit(metrics.get('grossMarginTTM', 'N/A'), 2)}")
            cols[2].metric(f"**Net Margin (TTM):**", f"{round_digit(metrics.get('netProfitMarginTTM', 'N/A'), 2)}")
            cols[3].metric(f"**Operating Margin (TTM):**", f"{round_digit(metrics.get('operatingMarginTTM', 'N/A'), 2)}")
            cols = st.columns([2, 2, 2, 2])
            cols[0].metric(f"**ROE (TTM):**", f"{round_digit(metrics.get('roeTTM', 'N/A'), 2)}")
            cols[1].metric(f"**ROA (TTM):**", f"{round_digit(metrics.get('roaTTM', 'N/A'), 2)}")
            cols[2].metric(f"**ROI (TTM):**", f"{round_digit(metrics.get('roiTTM', 'N/A'), 2)}")
            cols[3].metric(f"**5 Year EPS Growth:**", f"{round_digit(metrics.get('epsGrowth5Y', 'N/A'), 2)}")
            cols = st.columns([2, 2, 2, 2])
            cols[0].metric(f"**Beta:**", f"{round_digit(metrics.get('beta', 'N/A'), 2)}")
            cols[1].metric(f"**Debt/Equity Ratio:**", f"{round_digit(metrics.get('totalDebt/totalEquityQuarterly', 'N/A'), 2)}")
            cols[2].metric(f"**Current Ratio:**", f"{round_digit(metrics.get('currentRatioQuarterly', 'N/A'), 2)}")
            cols[3].metric(f"**Quick Ratio:**", f"{round_digit(metrics.get('quickRatioQuarterly', 'N/A'), 2)}")
            cols = st.columns([2, 2, 2, 2])
            cols[0].metric(f"**5D Return:**", f"{round_digit(metrics.get('5DayPriceReturnDaily', 'N/A'), 2)}")
            cols[1].metric(f"**13W Return:**", f"{round_digit(metrics.get('13WeekPriceReturnDaily', 'N/A'), 2)}")
            cols[2].metric(f"**26W Return:**", f"{round_digit(metrics.get('26WeekPriceReturnDaily', 'N/A'), 2)}")
            cols[3].metric(f"**52W Return:**", f"{round_digit(metrics.get('52WeekPriceReturnDaily', 'N/A'), 2)}")
            cols = st.columns([2, 2, 2, 2])
            cols[0].metric(f"**Dividend Yield:**", f"{round_digit(metrics.get('currentDividendYieldTTM', 'N/A'), 2)}")
            cols[1].metric(f"**P/CF Ratio:**", f"{round_digit(metrics.get('pcfShareTTM', 'N/A'), 2)}")
            cols[2].metric(f"**Net Profit Margin (TTM):**", f"{round_digit(metrics.get('netProfitMarginTTM', 'N/A'), 2)}")
            cols[3].metric(f"**Dividend Payout Ratio:**", f"{round_digit(metrics.get('payoutRatioTTM', 'N/A'), 2)}")


def indicator_system():
    """Manages adding and displaying technical indicators using session state only."""
    if 'indicators' not in st.session_state:
        st.session_state.indicators = []
    c1, c2, c3 = st.columns([2, 1, 1])
    ind_type = c1.selectbox("Indicator", ["SMA", "EMA", "MACD","RSI", "STO", "OBV", "ATR", "BOB", "ADX"], key="ind_type", label_visibility="collapsed")
    ind_period = c2.number_input("Period", min_value=2, max_value=200, value=20, key="ind_period", label_visibility="collapsed")
    if c3.button("Add", key="add_indicator", use_container_width=True):
        ind_tuple = (ind_type, int(ind_period))
        if ind_tuple not in st.session_state.indicators:
            st.session_state.indicators.append(ind_tuple)
            st.rerun()
    if st.session_state.indicators:
        st.markdown("Applied Indicators:")
        num_cols = min(len(st.session_state.indicators), 8)
        cols = st.columns(num_cols)
        for i, ind in enumerate(list(st.session_state.indicators)):
            with cols[i % num_cols]:
                ind_label = f"{ind[0]}-{ind[1]}"
                st.markdown(f"<div style='background:#2c2f36; padding:0.4rem; border-radius:0.5rem; text-align:center; margin-bottom:0.5rem;'>{ind_label}</div>", unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{ind_label}", use_container_width=True):
                    st.session_state.indicators.remove(ind)
                    st.rerun()
                    

COLOR_PALETTE = ['#ffd700', '#ff4081', '#7c4dff', '#00bcd4', '#00e676', '#ffb300', '#e040fb', '#29b6f6', '#ff1744', '#00e5ff']

def render_graph(symbol, show, interval, period, chart_type, realtime):
    """Renders the stock chart with technical indicators from session state."""
    if not show:
        st.warning("Graph display is turned off. Check the 'Show Graph' box to see the chart.")
        return
    try:
        if chart_type == "Candlestick":
            plot.candlestick(symbol, period, interval)
        else:
            plot.graph(symbol, period, interval)

        if realtime and is_market_open() == "True":
            st.caption("🟢 Real-time mode enabled. Auto-refreshing every 10 seconds.")
        elif realtime:
            str = is_market_open()
            st.caption("⚪ Real-time mode enabled, but " + str)
    except Exception as e:
        st.error(f"An error occurred while rendering the chart: {e}")