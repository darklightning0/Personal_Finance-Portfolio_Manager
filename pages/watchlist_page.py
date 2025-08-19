
import streamlit as st
import theme
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from streamlit_extras.stylable_container import stylable_container
from transformers import pipeline
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json
from utility import (
    get_symbol_list, get_stock_profile, get_stock_quote, get_stock_metrics, get_stock_lookup,
    search_component, indicator_system, is_market_open, display_toast_messages, create_financial_metrics_display, create_trading_interface,
    get_stock_news, graph, correlation_heatmap, candlestick, get_insider_transactions, get_insider_sentiment, get_recommendation_trends, get_eps_calendar
)

theme.custom_theme()

@st.cache_data(ttl=300)
def cached_stock_data(symbol):
    """Cache stock data to avoid repeated API calls"""
    profile = get_stock_profile(symbol)
    quote = get_stock_quote(symbol)
    metrics = get_stock_metrics(symbol)
    lookup = get_stock_lookup(symbol)
    return profile, quote, metrics, lookup

@st.cache_data(ttl=600)
def cached_insider_data(symbol, start_str, end_str):
    """Cache insider transactions and sentiment data"""
    transactions = get_insider_transactions(symbol, start_str, end_str)
    sentiment = get_insider_sentiment(symbol, start_str, end_str)
    return transactions, sentiment

@st.cache_data(ttl=600)
def cached_news_data(symbol, start_str, end_str):
    """Cache news data"""
    return get_stock_news(symbol, start_str, end_str)

@st.cache_data(ttl=1800)
def cached_recommendation_data(symbol):
    """Cache recommendation trends data"""
    return get_recommendation_trends(symbol)

@st.cache_data(ttl=1800)
def cached_eps_calendar_data(symbol=None, early_date=None, late_date=None):
    """Cache EPS calendar data and handle JSON parsing"""
    raw_data = get_eps_calendar(symbol=symbol, early_date=early_date, late_date=late_date)
    
    if isinstance(raw_data, str):
        try:
            parsed_data = json.loads(raw_data)
            if isinstance(parsed_data, dict) and 'earningsCalendar' in parsed_data:
                return parsed_data['earningsCalendar']
            elif isinstance(parsed_data, list):
                return parsed_data
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []
    elif isinstance(raw_data, dict):
        if 'earningsCalendar' in raw_data:
            return raw_data['earningsCalendar']
        else:
            return []
    elif isinstance(raw_data, list):
        return raw_data
    else:
        return []

def create_watchlist_button(symbol, profile):
    if 'watchlist_stocks' not in st.session_state:
        st.session_state.watchlist_stocks = []
    
    is_in_watchlist = symbol in st.session_state.watchlist_stocks
    
    with stylable_container(
        key=f"watchlist_btn_{symbol}",
        css_styles=f"""
        button {{
            background-color: {'#ff4444' if is_in_watchlist else '#00C851'};
            color: white;
            border: none;
            border-radius: 4px;
            width: 40px;
            height: 40px;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        button:hover {{
            background-color: {'#cc0000' if is_in_watchlist else '#007E33'};
        }}
        """
    ):
        button_text = "remove" if is_in_watchlist else "add"
        if st.button(label="" ,key=f"watchlist_toggle_{symbol}", icon=f":material/{button_text}:"):
            if is_in_watchlist:
                st.session_state.watchlist_stocks.remove(symbol)
                st.toast(f"Removed {symbol} from watchlist", icon="➖")
            else:
                st.session_state.watchlist_stocks.append(symbol)
                st.toast(f"Added {symbol} to watchlist", icon="➕")
            st.rerun()

def create_enhanced_info_display(symbol, profile, quote, metrics, lookup_data):
    logo_col, name_col, watchlist_col = st.columns([1, 8, 1])
    
    with logo_col:
        try:
            if profile.get('logo'):
                st.image(profile.get('logo'), width=80)
        except:
            st.write("📊")
    
    with name_col:
        st.markdown(f"### {profile.get('name', symbol)} ({symbol})")
    
    with watchlist_col:
        create_watchlist_button(symbol, profile)
    
    cols = st.columns([2, 2, 2, 2])
    cols[0].markdown(f"**Sector:**<br><span style='font-size: 1.5em; color:#ffe300;'>{profile.get('finnhubIndustry', 'N/A')}</span>", unsafe_allow_html=True)
    cols[1].markdown(f"**Stock Symbol:**<br><span style='font-size: 1.5em; color:#ffe300;'>{lookup_data.get('type', 'N/A')}</span>", unsafe_allow_html=True)
    cols[2].markdown(f"**Exchange:**<br><span style='font-size: 1.5em; color:#ffe300;'>{profile.get('exchange', 'N/A')}</span>", unsafe_allow_html=True)
    cols[3].markdown(f"**Market Cap (M):**<br><span style='font-size: 1.5em; color:#ffe300;'>${metrics.get('marketCapitalization', 'N/A')}</span>", unsafe_allow_html=True)
    
    cols = st.columns([2, 2, 2, 2])
    from utility import round_digit
    cols[0].metric("**Last Price:**", f"${round_digit(quote.get('c', 'N/A'), 2)}", delta=f"{round_digit(quote.get('d', 'N/A'), 2)} ({round_digit(quote.get('dp', 'N/A'), 2)}%)", delta_color="normal")
    cols[1].metric("**Open Price:**", f"${quote.get('o', 'N/A')}")
    cols[2].metric("**Highest Price:**", f"${quote.get('h', 'N/A')}")
    cols[3].metric("**Lowest Price:**", f"${quote.get('l', 'N/A')}")

def get_transaction_type_info(change, transaction_code):
    """Determine transaction type and color based on change value and transaction code"""
    if change > 0:
        return "BUY", "#00C851", "📈"
    elif change < 0:
        return "SELL", "#ff4444", "📉"
    else:
        return "NO CHANGE", "#ffbb33", "➖"

def format_currency(value):
    """Format currency values with proper formatting"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"${value:,.2f}"

def format_shares(value):
    """Format share values with comma separation"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"{int(value):,}"

def format_change(value):
    """Format change values with proper sign and formatting"""
    if pd.isna(value) or value == 0:
        return "0"
    sign = "+" if value > 0 else ""
    return f"{sign}{int(value):,}"

def create_insider_transactions_display(symbol, start_date, end_date, fetch_data):
    """Create insider transactions section with optimized data handling"""
    
    if start_date > end_date:
        st.error("Start date cannot be after end date!")
        return
    
    insider_key = f"insider_data_{symbol}"
    
    if insider_key not in st.session_state:
        st.session_state[insider_key] = None
    
    if fetch_data or st.session_state[insider_key] is None:
        with st.spinner("Fetching insider transactions data..."):
            try:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                transactions_data, _ = cached_insider_data(symbol, start_str, end_str)
                
                if transactions_data and transactions_data.get('data') and len(transactions_data['data']) > 0:
                    st.session_state[insider_key] = transactions_data['data']
                    st.success(f"Found {len(transactions_data['data'])} transactions")
                else:
                    st.warning("No insider transactions found for the selected date range")
                    st.session_state[insider_key] = []
                    return
                    
            except Exception as e:
                st.error(f"Error fetching insider transactions: {str(e)}")
                return
    
    transactions_data = st.session_state[insider_key]
    if not transactions_data or len(transactions_data) == 0:
        st.info("No insider transactions data available. Click 'Fetch Data' to load transactions.")
        return
    
    st.markdown(f"**Total Transactions:** {len(transactions_data)}")
    
    buy_data = [(t.get('change', 0), t.get('transactionPrice', 0)) for t in transactions_data if t.get('change', 0) > 0]
    sell_data = [(abs(t.get('change', 0)), t.get('transactionPrice', 0)) for t in transactions_data if t.get('change', 0) < 0]
    
    total_buy_shares = sum(change for change, _ in buy_data)
    total_buy_value = sum(change * price for change, price in buy_data if price)
    total_sell_shares = sum(change for change, _ in sell_data)
    total_sell_value = sum(change * price for change, price in sell_data if price)
    
    st.markdown("### 📊 Transaction Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 Total Buys", format_shares(total_buy_shares), f"{format_currency(total_buy_value)}")
    
    with col2:
        st.metric("🔴 Total Sells", format_shares(total_sell_shares), f"{format_currency(total_sell_value)}")
    
    with col3:
        net_shares = total_buy_shares - total_sell_shares
        net_color = "normal" if net_shares >= 0 else "inverse"
        st.metric("📈 Net Shares", format_change(net_shares), delta_color=net_color)
    
    with col4:
        net_value = total_buy_value - total_sell_value
        st.metric("💰 Net Value", format_currency(net_value), delta_color="normal" if net_value >= 0 else "inverse")

    if len(transactions_data) > 0:
        st.markdown("### 🧠 Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Bullish Signals:**")
            if buy_data:
                st.success(f"• {len(buy_data)} buy transactions")
                buy_prices = [price for _, price in buy_data if price and price > 0]
                if buy_prices:
                    avg_buy_price = sum(buy_prices) / len(buy_prices)
                    st.success(f"• Average buy price: {format_currency(avg_buy_price)}")
                else:
                    st.info("• No valid price data for buy transactions")
            else:
                st.info("• No buy transactions in this period")
        
        with col2:
            st.markdown("**📉 Bearish Signals:**")
            if sell_data:
                st.error(f"• {len(sell_data)} sell transactions")
                sell_prices = [price for _, price in sell_data if price]
                if sell_prices:
                    avg_sell_price = sum(sell_prices) / len(sell_prices)
                    st.error(f"• Average sell price: {format_currency(avg_sell_price)}")
            else:
                st.info("• No sell transactions in this period")
    
    st.markdown("---")
    
    st.markdown("### 📋 Recent Transaction Details")
    
    sorted_transactions = sorted(transactions_data, key=lambda x: x.get('transactionDate', ''), reverse=True)[:10]
    
    for transaction in sorted_transactions:
        change = transaction.get('change', 0)
        price = transaction.get('transactionPrice', 0)
        transaction_value = abs(change) * price if change and price else 0
        transaction_type, color, emoji = get_transaction_type_info(change, transaction.get('transactionCode', ''))
        
        with st.container():
            st.markdown(
                f"<div style='background-color: {color}; padding: 8px; border-radius: 5px; margin: 5px 0;'>"
                f"<strong style='color: white;'>{emoji} {transaction_type} - {transaction.get('name', 'Unknown')}</strong>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**Transaction Date:**<br>{transaction.get('transactionDate', 'N/A')}", unsafe_allow_html=True)
                st.markdown(f"**Filing Date:**<br>{transaction.get('filingDate', 'N/A')}", unsafe_allow_html=True)
            
            with col2:
                change_color = "#00C851" if change > 0 else "#ff4444"
                st.markdown(
                    f"**Shares Changed:**<br><span style='color: {change_color}; font-weight: bold;'>"
                    f"{format_change(change)}</span>", 
                    unsafe_allow_html=True
                )
                st.markdown(f"**Shares After:**<br>{format_shares(transaction.get('share', 0))}", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**Price per Share:**<br>{format_currency(price)}", unsafe_allow_html=True)
                st.markdown(f"**Transaction Value:**<br>{format_currency(transaction_value)}", unsafe_allow_html=True)
            
            with col4:
                code = transaction.get('transactionCode', 'N/A')
                st.markdown(f"**Transaction Code:**<br>{code}", unsafe_allow_html=True)
                
                code_explanations = {
                    'S': 'Sale', 'P': 'Purchase', 'M': 'Exercise/Conversion',
                    'A': 'Grant/Award', 'D': 'Disposition', 'F': 'Payment of Exercise Price',
                    'I': 'Discretionary Transaction', 'X': 'Exercise of In-the-Money Option'
                }
                code_meaning = code_explanations.get(code, 'Unknown')
                st.markdown(f"**Code Meaning:**<br><em>{code_meaning}</em>", unsafe_allow_html=True)
            
            st.markdown("---")
    
    if len(transactions_data) > 10:
        st.info(f"Showing 10 most recent transactions out of {len(transactions_data)} total.")

@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model with caching"""
    try:
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        return finbert
    except:
        st.warning("Could not load FinBERT model. Sentiment analysis disabled.")
        return None

def analyze_sentiment(text, model):
    """Analyze sentiment of given text using FinBERT"""
    if not model or not text:
        return {"label": "UNKNOWN", "score": 0.0}
    
    try:
        result = model(text[:512])
        return result[0] if result else {"label": "UNKNOWN", "score": 0.0}
    except:
        return {"label": "UNKNOWN", "score": 0.0}

def get_sentiment_color(label, score):
    """Get color for sentiment display"""
    colors = {
        "positive": "#00C851", "negative": "#ff4444", 
        "neutral": "#ffbb33", "UNKNOWN": "#6c757d"
    }
    return colors.get(label.lower(), "#6c757d")

def get_sentiment_emoji(label):
    """Get emoji for sentiment"""
    emojis = {
        "positive": "📈", "negative": "📉",
        "neutral": "➖", "UNKNOWN": "❓"
    }
    return emojis.get(label.lower(), "❓")

def get_mspr_color_and_emoji(mspr):
    """Get color and emoji based on MSPR value"""
    if mspr >= 20:
        return "#00C851", "🚀"
    elif mspr >= 5:
        return "#7CB342", "📈"
    elif mspr >= -5:
        return "#ffbb33", "➖"
    elif mspr >= -20:
        return "#FF8A65", "📉"
    else:
        return "#ff4444", "💸"
    
def create_insider_sentiment_section(symbol, start_date, end_date):
    """Create insider sentiment analysis section with MSPR data"""
    st.markdown("### 🧠 Insider Sentiment Analysis (MSPR)")
    
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        _, sentiment_data = cached_insider_data(symbol, start_str, end_str)
        
        if sentiment_data and sentiment_data.get('data') and len(sentiment_data['data']) > 0:
            sentiment_records = sentiment_data['data']
            
            sentiment_records.sort(key=lambda x: (x.get('year', 0), x.get('month', 0)), reverse=True)
            
            latest_record = sentiment_records[0]
            latest_mspr = latest_record.get('mspr', 0)
            latest_change = latest_record.get('change', 0)
            
            color, emoji = get_mspr_color_and_emoji(latest_mspr)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"**Latest MSPR:**<br>"
                    f"<span style='color: {color}; font-size: 1.5em; font-weight: bold;'>"
                    f"{emoji} {latest_mspr:.2f}</span>", 
                    unsafe_allow_html=True
                )
            
            with col2:
                change_color = "#00C851" if latest_change >= 0 else "#ff4444"
                change_sign = "+" if latest_change >= 0 else ""
                st.markdown(
                    f"**Latest Net Change:**<br>"
                    f"<span style='color: {change_color}; font-size: 1.2em; font-weight: bold;'>"
                    f"{change_sign}{latest_change:,}</span>", 
                    unsafe_allow_html=True
                )
            
            with col3:
                avg_mspr = sum(r.get('mspr', 0) for r in sentiment_records) / len(sentiment_records)
                avg_color, avg_emoji = get_mspr_color_and_emoji(avg_mspr)
                st.markdown(
                    f"**Average MSPR:**<br>"
                    f"<span style='color: {avg_color}; font-size: 1.2em; font-weight: bold;'>"
                    f"{avg_emoji} {avg_mspr:.2f}</span>", 
                    unsafe_allow_html=True
                )
            
            with col4:
                total_change = sum(r.get('change', 0) for r in sentiment_records)
                total_color = "#00C851" if total_change >= 0 else "#ff4444"
                total_sign = "+" if total_change >= 0 else ""
                st.markdown(
                    f"**Total Net Change:**<br>"
                    f"<span style='color: {total_color}; font-size: 1.2em; font-weight: bold;'>"
                    f"{total_sign}{total_change:,}</span>", 
                    unsafe_allow_html=True
                )
            
            st.markdown("#### 📅 Monthly Sentiment Trend")
            
            for record in sentiment_records[:6]:
                year = record.get('year', 'N/A')
                month = record.get('month', 'N/A')
                mspr = record.get('mspr', 0)
                change = record.get('change', 0)
                
                color, emoji = get_mspr_color_and_emoji(mspr)
                
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_name = month_names[month] if isinstance(month, int) and 1 <= month <= 12 else str(month)
                
                with st.container():
                    st.markdown(
                        f"<div style='background-color: {color}20; border-left: 4px solid {color}; "
                        f"padding: 10px; margin: 5px 0; border-radius: 5px;'>"
                        f"<strong>{emoji} {month_name} {year}</strong><br>"
                        f"MSPR: <span style='color: {color}; font-weight: bold;'>{mspr:.2f}</span> | "
                        f"Net Change: <span style='color: {'#00C851' if change >= 0 else '#ff4444'}; font-weight: bold;'>"
                        f"{'+' if change >= 0 else ''}{change:,}</span>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
        
        else:
            st.warning("No insider sentiment data available for the selected date range.")
            
    except Exception as e:
        st.error(f"Error fetching insider sentiment data: {str(e)}")

def format_news_date(timestamp):
    """Format Unix timestamp to readable date"""
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    except:
        return "Unknown Date"

def create_news_section(symbol, start_date, end_date):
    """Create news section with optimized data handling"""
    
    if start_date > end_date:
        st.error("Start date cannot be after end date!")
        return
    
    news_key = f"news_data_{symbol}"
    page_key = f"news_page_{symbol}"
    
    if news_key not in st.session_state:
        st.session_state[news_key] = None
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=start_date,
            max_value=datetime.now(),
            key=f"news_start_{symbol}"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=end_date,
            max_value=datetime.now(),
            key=f"news_end_{symbol}"
        )
    
    with col3:
        st.markdown("&nbsp;")
        fetch_news = st.button(
            "📰 Fetch News", 
            key=f"fetch_news_{symbol}",
            use_container_width=True
        )
    
    if fetch_news:
        with st.spinner("Fetching news data..."):
            try:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                news_data = cached_news_data(symbol, start_str, end_str)
                
                if news_data and len(news_data) > 0:
                    st.session_state[news_key] = news_data
                    st.session_state[page_key] = 1
                    st.success(f"Found {len(news_data)} news articles")
                else:
                    st.warning("No news found for the selected date range")
                    return
                    
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
                return
    
    news_data = st.session_state[news_key]
    if not news_data or len(news_data) == 0:
        st.info("Click 'Fetch News' to load articles for the selected date range.")
        return
    
    sentiment_model = load_sentiment_model()
    if sentiment_model:
        st.markdown("### 📊 Sentiment Analysis Overview")
        
        sentiment_stats = {'POSITIVE': [], 'NEGATIVE': [], 'NEUTRAL': []}
        
        for article in news_data:
            summary = article.get('summary', '')
            if summary:
                sentiment = analyze_sentiment(summary, sentiment_model)
                label = sentiment.get('label', 'NEUTRAL').upper()
                score = sentiment.get('score', 0.0)
                
                if label in sentiment_stats:
                    sentiment_stats[label].append(score)
                else:
                    sentiment_stats['NEUTRAL'].append(score)
        
        total_articles = len(news_data)
        positive_count = len(sentiment_stats['POSITIVE'])
        negative_count = len(sentiment_stats['NEGATIVE'])
        neutral_count = len(sentiment_stats['NEUTRAL'])
        
        avg_positive = sum(sentiment_stats['POSITIVE']) / len(sentiment_stats['POSITIVE']) if sentiment_stats['POSITIVE'] else 0
        avg_negative = sum(sentiment_stats['NEGATIVE']) / len(sentiment_stats['NEGATIVE']) if sentiment_stats['NEGATIVE'] else 0
        avg_neutral = sum(sentiment_stats['NEUTRAL']) / len(sentiment_stats['NEUTRAL']) if sentiment_stats['NEUTRAL'] else 0
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric(
                label="📈 Positive",
                value=f"{positive_count} ({positive_count/total_articles*100:.1f}%)",
                delta=f"Avg Confidence: {avg_positive:.2f}" if positive_count > 0 else "No articles"
            )
        
        with stat_col2:
            st.metric(
                label="📉 Negative", 
                value=f"{negative_count} ({negative_count/total_articles*100:.1f}%)",
                delta=f"Avg Confidence: {avg_negative:.2f}" if negative_count > 0 else "No articles"
            )
        
        with stat_col3:
            st.metric(
                label="➖ Neutral",
                value=f"{neutral_count} ({neutral_count/total_articles*100:.1f}%)",
                delta=f"Avg Confidence: {avg_neutral:.2f}" if neutral_count > 0 else "No articles"
            )
        
        with stat_col4:
            if positive_count > negative_count:
                trend = "📈 Bullish"
                trend_color = "green"
            elif negative_count > positive_count:
                trend = "📉 Bearish"
                trend_color = "red"
            else:
                trend = "➖ Neutral"
                trend_color = "gray"
            
            st.metric(
                label="Overall Trend",
                value=trend,
                delta=f"Based on {total_articles} articles"
            )
        
        st.markdown("---")
    
    ITEMS_PER_PAGE = 5
    total_items = len(news_data)
    total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    current_page = st.session_state[page_key]
    
    st.markdown(f"**Total Articles:** {total_items} | **Page:** {current_page} of {total_pages}")
    
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    current_articles = news_data[start_idx:end_idx]
    
    for article in current_articles:
        with st.container():
            col_img, col_content = st.columns([1, 4])
            
            with col_img:
                if article.get('image'):
                    try:
                        st.image(article['image'], width=120, use_container_width=True)
                    except:
                        st.markdown("📰")
                else:
                    st.markdown("📰")
            
            with col_content:
                headline = article.get('headline', 'No headline available')
                url = article.get('url', '#')
                
                if url and url != '#':
                    st.markdown(f"**[{headline}]({url})**")
                else:
                    st.markdown(f"**{headline}**")
                
                summary = article.get('summary', 'No summary available')
                st.markdown(f"*{summary[:200]}{'...' if len(summary) > 200 else ''}*")
                
                if sentiment_model and summary:
                    sentiment = analyze_sentiment(summary, sentiment_model)
                    
                    sentiment_label = sentiment.get('label', 'UNKNOWN').upper()
                    sentiment_score = sentiment.get('score', 0.0)
                    sentiment_color = get_sentiment_color(sentiment_label, sentiment_score)
                    sentiment_emoji = get_sentiment_emoji(sentiment_label)
                    
                    st.markdown(
                        f"**Sentiment:** {sentiment_emoji} "
                        f"<span style='color: {sentiment_color}; font-weight: bold;'>"
                        f"{sentiment_label}</span> "
                        f"(Confidence: {sentiment_score:.2f})",
                        unsafe_allow_html=True
                    )
                
                col_source, col_time = st.columns([2, 2])
                with col_source:
                    source = article.get('source', 'Unknown Source')
                    st.caption(f"**Source:** {source}")
                
                with col_time:
                    try:
                        pub_time = datetime.fromtimestamp(article.get('datetime', 0))
                        st.caption(f"**Published:** {pub_time.strftime('%H:%M')}")
                    except:
                        st.caption("**Published:** Unknown")
            
            st.markdown("---")
    
    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if current_page > 1:
                if st.button("← Previous", key=f"prev_news_{symbol}"):
                    st.session_state[page_key] = current_page - 1
                    st.rerun()
        
        with col_info:
            st.markdown(f"<div style='text-align: center;'>Page {current_page} of {total_pages}</div>", 
                       unsafe_allow_html=True)
        
        with col_next:
            if current_page < total_pages:
                if st.button("Next →", key=f"next_news_{symbol}"):
                    st.session_state[page_key] = current_page + 1
                    st.rerun()

def create_recommendation_trends(symbol):
    """Creates and displays a stacked bar chart of analyst recommendation trends."""
    try:
        data_list = cached_recommendation_data(symbol)

        if not data_list:
            st.warning(f"No recommendation data found for {symbol}.")
            return

        data_restructured = {}
        for item in data_list:
            try:
                date_obj = datetime.strptime(item.get("period", ""), "%Y-%m-%d")
                month_year = date_obj.strftime("%b %Y")
                data_restructured[month_year] = {
                    'Strong Buy': item.get('strongBuy', 0),
                    'Buy': item.get('buy', 0),
                    'Hold': item.get('hold', 0),
                    'Sell': item.get('sell', 0),
                    'Strong Sell': item.get('strongSell', 0),
                }
            except ValueError:
                continue

        if not data_restructured:
            st.warning(f"No valid recommendation data found for {symbol}.")
            return

        recommendation_order = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
        colors = ['#1D7D41', '#419E5C', '#BF9344', '#E65545', '#993404']

        months = list(data_restructured.keys())
        fig_data = []

        for i, rec_type in enumerate(recommendation_order):
            rec_counts = [data_restructured[month][rec_type] for month in months]

            fig_data.append(go.Bar(
                name=rec_type,
                x=months,
                y=rec_counts,
                marker_color=colors[i],
                text=rec_counts,
                textposition='auto',
                textfont=dict(color='white'),
                hoverinfo='text',
                hovertext=[f'{rec_type}: {count}' for count in rec_counts]
            ))

        fig = go.Figure(data=fig_data)

        fig.update_layout(
            title_text=f'{symbol} Recommendation Trends',
            barmode='stack',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.5,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(title_text='Month', showgrid=False),
            yaxis=dict(title_text='# Analysts', showgrid=False),
            margin=dict(l=50, r=50, t=50, b=150)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating recommendation trends: {str(e)}")

def get_eps_performance_color_and_emoji(eps_actual, eps_estimate):
    """Get color and emoji based on EPS performance vs estimate"""
    if eps_actual is None or eps_estimate is None:
        return "#6c757d", "❓", "No Data"
    
    try:
        eps_actual = float(eps_actual)
        eps_estimate = float(eps_estimate)
        
        if eps_actual > eps_estimate:
            return "#00C851", "🚀", "Beat Estimate"
        elif eps_actual == eps_estimate:
            return "#ffbb33", "🎯", "Met Estimate"
        else:
            return "#ff4444", "📉", "Missed Estimate"
    except (ValueError, TypeError):
        return "#6c757d", "❓", "Invalid Data"

def get_revenue_performance_color_and_emoji(revenue_actual, revenue_estimate):
    """Get color and emoji based on Revenue performance vs estimate"""
    if revenue_actual is None or revenue_estimate is None:
        return "#6c757d", "❓", "No Data"
    
    try:
        revenue_actual = float(revenue_actual)
        revenue_estimate = float(revenue_estimate)
        
        if revenue_actual > revenue_estimate:
            return "#00C851", "🚀", "Beat Estimate"
        elif revenue_actual == revenue_estimate:
            return "#ffbb33", "🎯", "Met Estimate"
        else:
            return "#ff4444", "📉", "Missed Estimate"
    except (ValueError, TypeError):
        return "#6c757d", "❓", "Invalid Data"

def format_eps_value(value):
    """Format EPS value for display"""
    if value is None:
        return "N/A"
    try:
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_revenue_value(value):
    """Format revenue value for display"""
    if value is None:
        return "N/A"
    try:
        value_float = float(value)
        if value_float >= 1e9:
            return f"${value_float / 1e9:.2f}B"
        elif value_float >= 1e6:
            return f"${value_float / 1e6:.2f}M"
        elif value_float >= 1e3:
            return f"${value_float / 1e3:.2f}K"
        else:
            return f"${value_float:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def create_eps_calendar_section():
    """Create EPS calendar section with filtering and visualization"""
    st.markdown("### 📅 EPS Calendar & Earnings Analysis")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=15),
            key="eps_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() + timedelta(days=15),
            key="eps_end_date"
        )
    
    with col3:
        symbol_filter = st.text_input(
            "Filter by Symbol (optional)",
            value="",
            key="eps_symbol_filter",
            placeholder="e.g., AAPL"
        ).upper().strip()
    
    with col4:
        st.markdown("&nbsp;")
        fetch_eps = st.button(
            "📊 Fetch EPS Data",
            key="fetch_eps_data",
            use_container_width=True
        )
    
    if start_date > end_date:
        st.error("Start date cannot be after end date!")
        return
    
    eps_key = "eps_calendar_data"
    
    if eps_key not in st.session_state:
        st.session_state[eps_key] = None
    
    if fetch_eps or st.session_state[eps_key] is None:
        with st.spinner("Fetching EPS calendar data..."):
            try:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                filter_symbol = symbol_filter if symbol_filter else None
                
                eps_data = cached_eps_calendar_data(
                    symbol=filter_symbol,
                    early_date=start_str,
                    late_date=end_str
                )
                
                if eps_data and len(eps_data) > 0:
                    st.session_state[eps_key] = eps_data
                    st.success(f"Found {len(eps_data)} earnings records")
                else:
                    st.warning("No EPS data found for the selected criteria")
                    st.session_state[eps_key] = []
                    return
                    
            except Exception as e:
                st.error(f"Error fetching EPS data: {str(e)}")
                return
    
    eps_data = st.session_state[eps_key]
    if not eps_data or len(eps_data) == 0:
        st.info("Click 'Fetch EPS Data' to load earnings calendar.")
        return
    
    filtered_data = eps_data
    if symbol_filter:
        filtered_data = [item for item in eps_data if isinstance(item, dict) and item.get('symbol', '').upper() == symbol_filter]
    
    if not filtered_data:
        if symbol_filter:
            st.warning(f"No data found for symbol: {symbol_filter}")
        else:
            st.warning("No valid EPS data found.")
        return
    
    st.markdown(f"**Total Records:** {len(filtered_data)}")
    
    performance_stats = {'beat': 0, 'met': 0, 'missed': 0, 'no_data': 0}
    revenue_stats = {'beat': 0, 'met': 0, 'missed': 0, 'no_data': 0}
    
    for item in filtered_data:
        if not isinstance(item, dict):
            continue
            
        eps_actual = item.get('epsActual')
        eps_estimate = item.get('epsEstimate')
        revenue_actual = item.get('revenueActual')
        revenue_estimate = item.get('revenueEstimate')
        
        _, _, eps_status = get_eps_performance_color_and_emoji(eps_actual, eps_estimate)
        _, _, revenue_status = get_revenue_performance_color_and_emoji(revenue_actual, revenue_estimate)
        
        if eps_status == "Beat Estimate":
            performance_stats['beat'] += 1
        elif eps_status == "Met Estimate":
            performance_stats['met'] += 1
        elif eps_status == "Missed Estimate":
            performance_stats['missed'] += 1
        else:
            performance_stats['no_data'] += 1
        
        if revenue_status == "Beat Estimate":
            revenue_stats['beat'] += 1
        elif revenue_status == "Met Estimate":
            revenue_stats['met'] += 1
        elif revenue_status == "Missed Estimate":
            revenue_stats['missed'] += 1
        else:
            revenue_stats['no_data'] += 1
    
    st.markdown("### 📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🚀 EPS Beats",
            value=f"{performance_stats['beat']} ({performance_stats['beat']/len(filtered_data)*100:.1f}%)",
            delta="Bullish Signal"
        )
    
    with col2:
        st.metric(
            label="📉 EPS Misses",
            value=f"{performance_stats['missed']} ({performance_stats['missed']/len(filtered_data)*100:.1f}%)",
            delta="Bearish Signal"
        )
    
    with col3:
        st.metric(
            label="🚀 Revenue Beats",
            value=f"{revenue_stats['beat']} ({revenue_stats['beat']/len(filtered_data)*100:.1f}%)",
            delta="Bullish Signal"
        )
    
    with col4:
        st.metric(
            label="📉 Revenue Misses",
            value=f"{revenue_stats['missed']} ({revenue_stats['missed']/len(filtered_data)*100:.1f}%)",
            delta="Bearish Signal"
        )
    
    st.markdown("---")
    
    st.markdown("### 📋 Earnings Details")
    
    valid_data = [item for item in filtered_data if isinstance(item, dict)]
    sorted_eps_data = sorted(valid_data, key=lambda x: x.get('date', ''), reverse=True)
    
    for item in sorted_eps_data[:20]:
        symbol = item.get('symbol', 'N/A')
        date = item.get('date', 'N/A')
        hour = item.get('hour', 'N/A')
        quarter = item.get('quarter', 'N/A')
        year = item.get('year', 'N/A')
        
        eps_actual = item.get('epsActual')
        eps_estimate = item.get('epsEstimate')
        revenue_actual = item.get('revenueActual')
        revenue_estimate = item.get('revenueEstimate')
        
        eps_color, eps_emoji, eps_status = get_eps_performance_color_and_emoji(eps_actual, eps_estimate)
        revenue_color, revenue_emoji, revenue_status = get_revenue_performance_color_and_emoji(revenue_actual, revenue_estimate)
        
        with st.container():
            st.markdown(
                f"<div style='background: linear-gradient(90deg, {eps_color}20, {revenue_color}20); "
                f"border-left: 4px solid {eps_color}; padding: 12px; margin: 8px 0; border-radius: 8px;'>"
                f"<h4 style='margin: 0; color: white;'>{symbol} - Q{quarter} {year} Earnings</h4>"
                f"<p style='margin: 4px 0; color: #ccc;'>📅 {date} | ⏰ {hour.upper() if hour else 'N/A'}</p>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"**EPS Performance:**<br>"
                    f"<span style='color: {eps_color}; font-size: 1.2em; font-weight: bold;'>"
                    f"{eps_emoji} {eps_status}</span>", 
                    unsafe_allow_html=True
                )
                st.markdown(f"**Actual:** {format_eps_value(eps_actual)}")
                st.markdown(f"**Estimate:** {format_eps_value(eps_estimate)}")
            
            with col2:
                if eps_actual is not None and eps_estimate is not None:
                    try:
                        eps_surprise = float(eps_actual) - float(eps_estimate)
                        eps_surprise_pct = (eps_surprise / float(eps_estimate)) * 100 if float(eps_estimate) != 0 else 0
                        surprise_color = "#00C851" if eps_surprise >= 0 else "#ff4444"
                        surprise_sign = "+" if eps_surprise >= 0 else ""
                        
                        st.markdown(f"**EPS Surprise:**")
                        st.markdown(
                            f"<span style='color: {surprise_color}; font-weight: bold;'>"
                            f"{surprise_sign}{eps_surprise:.2f} ({surprise_sign}{eps_surprise_pct:.1f}%)</span>", 
                            unsafe_allow_html=True
                        )
                    except (ValueError, TypeError, ZeroDivisionError):
                        st.markdown(f"**EPS Surprise:** N/A")
                else:
                    st.markdown(f"**EPS Surprise:** N/A")
            
            with col3:
                st.markdown(
                    f"**Revenue Performance:**<br>"
                    f"<span style='color: {revenue_color}; font-size: 1.2em; font-weight: bold;'>"
                    f"{revenue_emoji} {revenue_status}</span>", 
                    unsafe_allow_html=True
                )
                st.markdown(f"**Actual:** {format_revenue_value(revenue_actual)}")
                st.markdown(f"**Estimate:** {format_revenue_value(revenue_estimate)}")
            
            with col4:
                if revenue_actual is not None and revenue_estimate is not None:
                    try:
                        revenue_surprise = float(revenue_actual) - float(revenue_estimate)
                        revenue_surprise_pct = (revenue_surprise / float(revenue_estimate)) * 100 if float(revenue_estimate) != 0 else 0
                        rev_surprise_color = "#00C851" if revenue_surprise >= 0 else "#ff4444"
                        rev_surprise_sign = "+" if revenue_surprise >= 0 else ""
                        
                        st.markdown(f"**Revenue Surprise:**")
                        st.markdown(
                            f"<span style='color: {rev_surprise_color}; font-weight: bold;'>"
                            f"{rev_surprise_sign}{format_revenue_value(abs(revenue_surprise))} "
                            f"({rev_surprise_sign}{revenue_surprise_pct:.1f}%)</span>", 
                            unsafe_allow_html=True
                        )
                    except (ValueError, TypeError, ZeroDivisionError):
                        st.markdown(f"**Revenue Surprise:** N/A")
                else:
                    st.markdown(f"**Revenue Surprise:** N/A")
            
            st.markdown("---")
    
    if len(sorted_eps_data) > 20:
        st.info(f"Showing 20 most recent records out of {len(sorted_eps_data)} total.")

def create_watchlist_section():
    """Optimized watchlist section with better data handling"""
    if 'watchlist_stocks' not in st.session_state or not st.session_state.watchlist_stocks:
        st.info("📊 No stocks in watchlist. Search for stocks and add them to your watchlist.")
        return
    
    st.subheader("📊 Your Watchlist")
    
    watchlist_data = []
    with st.spinner("Loading watchlist data..."):
        for symbol in st.session_state.watchlist_stocks:
            try:
                quote = get_stock_quote(symbol)
                if quote:
                    current_price = quote.get('c', 0)
                    change = quote.get('d', 0)
                    change_pct = quote.get('dp', 0)
                    
                    watchlist_data.append({
                        'Symbol': symbol,
                        'Price': f"${current_price:.2f}" if current_price else "N/A",
                        'Change': f"{change:+.2f}" if change else "0.00",
                        'Change %': f"{change_pct:+.2f}%" if change_pct else "0.00%"
                    })
                else:
                    watchlist_data.append({
                        'Symbol': symbol,
                        'Price': "N/A",
                        'Change': "N/A",
                        'Change %': "N/A"
                    })
            except:
                watchlist_data.append({
                    'Symbol': symbol,
                    'Price': "N/A",
                    'Change': "N/A",
                    'Change %': "N/A"
                })
    
    if watchlist_data:
        df = pd.DataFrame(watchlist_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("**Remove stocks:**")
        
        if st.button("Remove All Stocks", key="remove_all_stocks", 
                    use_container_width=True, type="primary", 
                    help="Clear all stocks from watchlist"):
            st.session_state.watchlist_stocks.clear()
            if 'watchlist_data' in st.session_state:
                st.session_state.watchlist_data = []
            st.rerun()
        
        cols_per_row = 4
        symbols = st.session_state.watchlist_stocks.copy()
        for i in range(0, len(symbols), cols_per_row):
            remove_cols = st.columns(cols_per_row)
            for j, symbol in enumerate(symbols[i:i+cols_per_row]):
                with remove_cols[j]:
                    if st.button(f"✕ {symbol}", key=f"remove_{symbol}", 
                                use_container_width=True, type="secondary"):
                        st.session_state.watchlist_stocks.remove(symbol)
                        st.rerun()

def create_watchlist_graph_controls():
    """Optimized graph controls"""
    if 'watchlist_stocks' not in st.session_state:
        st.session_state.watchlist_stocks = []

    if 'current_chart_type' not in st.session_state:
        st.session_state.current_chart_type = "Line"
    
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
                "Interval", options=list(VALID_PERIODS.keys()), 
                index=0, key="watchlist_interval_select"
            )
            period_options = VALID_PERIODS[interval]
            period = st.selectbox(
                "Period", options=period_options, 
                index=3 if interval == "1d" else len(period_options) - 1, 
                key="watchlist_period_select"
            )
            chart_type = st.selectbox(
                "Chart Type", 
                ["Line", "Candlestick", "Correlation Heatmap"], 
                index=["Line", "Candlestick", "Correlation Heatmap"].index(
                    st.session_state.current_chart_type
                ),
                key="watchlist_chart_type_select"
            )
            
            if chart_type != st.session_state.current_chart_type:
                st.session_state.current_chart_type = chart_type
                st.rerun()
                
        with col2:
            st.markdown("##### Technical Indicators")
            indicator_system()
        
        st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        show_graph = c3.checkbox("Show Graph", value=True, key="watchlist_show_graph")
        realtime = c4.checkbox("Real-Time", value=False, key="watchlist_realtime")

    return interval, period, chart_type, show_graph, realtime

def render_watchlist_graph(symbols, show, interval, period, chart_type, realtime):
    """Optimized graph rendering"""
    if not show:
        st.warning("Graph display is turned off. Check the 'Show Graph' box to see the chart.")
        return
    
    if not symbols:
        st.info("Add stocks to your watchlist to see the combined chart.")
        return
    
    try:
        if chart_type == "Candlestick":
            if len(symbols) == 1:
                candlestick(symbols[0], period, interval)
            else:
                st.error("📊 Only one stock can be shown using candlesticks at a time.")
        elif chart_type == "Correlation Heatmap":
            correlation_heatmap(symbols, period)
        else:
            graph(symbols, period, interval)

        if realtime:
            market_status = is_market_open()
            if market_status == "True":
                st.caption("🟢 Real-time mode enabled. Auto-refreshing every 15 seconds.")
            else:
                st.caption(f"⚪ Real-time mode enabled, but {market_status}")
                
    except Exception as e:
        st.error(f"An error occurred while rendering the chart: {e}")

def display_chart_info(symbols, chart_type):
    """Display chart information"""
    if not symbols:
        return
        
    num_symbols = len(symbols)
    
    if chart_type == "Line":
        st.caption(f"📈 Line chart displaying {num_symbols} stock{'s' if num_symbols > 1 else ''}: {', '.join(symbols)}")
    elif chart_type == "Correlation Heatmap":
        st.caption(f"📈 Correlation Heatmap displaying {num_symbols} stock{'s' if num_symbols > 1 else ''}: {', '.join(symbols)}")
    else:
        st.caption(f"🕯️ Candlestick chart for {symbols[0]}")

def create_stock_analysis_tabs(symbol, profile, quote, metrics, lookup_data):
    """Create organized tabs for stock analysis sections"""
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🧮 Financial Metrics", 
        "📊 Analyst Recommendations", 
        "🏢 Insider Activity", 
        "📰 News & Sentiment",
        "📅 EPS Calendar",
        "💼 Trading",
        "📈 Chart"
    ])
    
    with tab1:
        st.markdown("### 🧮 Comprehensive Financial Metrics")
        create_financial_metrics_display(symbol, metrics)
    
    with tab2:
        st.markdown("### 📊 Analyst Recommendation Trends")
        create_recommendation_trends(symbol)
    
    with tab3:
        st.markdown("### 🏢 Insider Transactions & Sentiment")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=90), 
                max_value=datetime.now(),
                key=f"insider_start_{symbol}"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
                key=f"insider_end_{symbol}"
            )
        
        with col3:
            st.markdown("&nbsp;")
            fetch_data = st.button(
                "📊 Fetch Data", 
                key=f"fetch_insider_{symbol}",
                use_container_width=True
            )
        
        insider_tab1, insider_tab2 = st.tabs(["📊 Sentiment (MSPR)", "📋 Transactions"])
        
        with insider_tab1:
            create_insider_sentiment_section(symbol, start_date, end_date)
        
        with insider_tab2:
            create_insider_transactions_display(symbol, start_date, end_date, fetch_data)
    
    with tab4:
        st.markdown("### 📰 News & Sentiment Analysis")
        default_start = datetime.now() - timedelta(days=30)
        default_end = datetime.now()
        create_news_section(symbol, default_start, default_end)
    
    with tab5:
        create_eps_calendar_section()
    
    with tab6:
        st.markdown("### 💼 Trading Interface")
        create_trading_interface(symbol=symbol)

    with tab7:
        st.markdown("### 📈 Chart")
        interval, period, chart_type, show_graph, realtime = create_watchlist_graph_controls()
        watchlist_symbols = st.session_state.get('watchlist_stocks', [])
        if watchlist_symbols and show_graph:
            display_chart_info(watchlist_symbols, chart_type)

        if realtime and is_market_open() == "True":
            st_autorefresh(interval=15000, limit=1000, key="watchlist_rt_refresh")

        render_watchlist_graph(
            symbols=watchlist_symbols,
            show=show_graph,
            interval=interval,
            period=period,
            chart_type=chart_type,
            realtime=realtime
        )

st.title("Watchlist")

if "indicators" not in st.session_state:
    st.session_state.indicators = []

if 'watchlist_stocks' not in st.session_state:
    st.session_state.watchlist_stocks = []

symbols = get_symbol_list()
selected_symbol = search_component(symbols)

if selected_symbol:
    profile, quote, metrics, lookup = cached_stock_data(selected_symbol)
    
    if not lookup or not lookup.get('result'):
        st.error("Could not retrieve lookup data for this stock.")
    else:
        lookup_data = lookup['result'][0]
        
        if not profile or not quote:
            st.error("Could not retrieve all data for this stock.")
        else:
            display_toast_messages()
            
            create_enhanced_info_display(selected_symbol, profile, quote, metrics, lookup_data)
            
            create_stock_analysis_tabs(selected_symbol, profile, quote, metrics, lookup_data)

create_watchlist_section()