import streamlit as st

def custom_theme():
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            background-color: #111215 !important;
            color: #fff !important;
            font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
            transition: background 0.3s;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
            margin-top: 2rem;
        }
        .dashboard-header {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            color: #fff;
        }
        .header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.2rem;
            gap: 1rem;
        }
        .info-bar-inline {
            display: flex;
            align-items: center;
            background: #181A20;
            border-radius: 0.7rem;
            padding: 0.35rem 1rem;
            font-size: 1rem;
            color: #fff;
            box-shadow: 0 1px 8px 0 #00000022;
            opacity: 0.92;
            gap: 0.7rem;
            min-width: 220px;
            max-width: 400px;
        }
        .info-bar-inline .weather-emoji {
            font-size: 1.2rem;
            margin-right: 0.2rem;
            vertical-align: middle;
        }
        .info-bar-inline .weather-detail {
            color: #2196f3;
            font-weight: 600;
            margin-right: 0.2rem;
            white-space: nowrap;
        }
        .info-bar-inline .date-detail {
            color: #fff;
            font-weight: 400;
            opacity: 0.8;
            margin-right: 0.4rem;
            white-space: nowrap;
        }
        @media (max-width: 600px) {
            .header-row { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
            .info-bar-inline { width: 100%; min-width: 0; max-width: 100%; font-size: 0.95rem; }
        }
        /* Neon class for future use */
        .neon {
            color: #00eaff;
            text-shadow: 0 0 8px #00eaff, 0 0 16px #00eaff44;
            font-weight: 800;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)