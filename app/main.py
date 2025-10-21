import streamlit as st
from app.config import Config
from app.stock_analysis import StockAnalysisDashboard
from app.ai_assistant import AIFinancialAssistant
from app.stock_comparison import StockComparisonAgent
from app.portfolio_simulator import PortfolioSimulator
from app.backtesting import BacktestingAgent
import yfinance as yf
import pandas as pd
from app.utils import fetch_top_movers

def get_sp500_data():
    sp500 = yf.Ticker("^GSPC")
    info = sp500.info
    price = info.get('regularMarketPrice', info.get('previousClose', 'N/A'))
    change_percent = info.get('regularMarketChangePercent', 0)
    return price, change_percent

def main():
    st.set_page_config(
        page_title="AI Investment Agent 🤖",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .news-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #333;
        transition: all 0.3s;
    }
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Add logo to sidebar
    st.sidebar.image("Stock_Logo.png", width=200)
    
    # Define page options
    page_options = ["Home", "Stock Analysis", "AI Assistant", "Stock Comparison", "Backtesting"]
    
    # Create radio button with session state synchronization
    page = st.sidebar.radio(
        "Go to", 
        page_options,
        index=page_options.index(st.session_state.page)
    )
    
    # Update session state when radio changes
    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()
    
    # Main content area
    with st.container():
        if st.session_state.page == "Home":
            display_home()
        elif st.session_state.page == "Stock Analysis":
            StockAnalysisDashboard().run()
        elif st.session_state.page == "AI Assistant":
            ai_assistant = AIFinancialAssistant()
            ai_assistant.run()
        elif st.session_state.page == "Stock Comparison":
            StockComparisonAgent().run()
        elif st.session_state.page == "Backtesting":
            backtesting = BacktestingAgent()
            backtesting.run()

    # Footer
    # st.markdown("---")
    # st.markdown("Developed by PiSpace.co. 2024 | [Terms of Service](/) | [Privacy Policy](/)")

def display_home():
    # Hero Section
    st.title("AI Investment Agent")
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Make smarter investment decisions with AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main Features
    st.subheader("🛠️ Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Stock Analysis")
        st.write("""
        - Technical analysis with indicators
        - Price predictions using ML
        - Financial metrics and ratios
        """)
        if st.button("Analyze Stocks", key="analyze"):
            st.session_state.page = "Stock Analysis"
            st.rerun()

        st.markdown("### 🤖 AI Assistant")
        st.write("""
        - Get instant market insights
        - Ask financial questions
        - Analyze charts and data
        """)
        if st.button("Chat with Pi", key="chat"):
            st.session_state.page = "AI Assistant"
            st.rerun()

    with col2:
        st.markdown("### 🔄 Stock Comparison")
        st.write("""
        - Compare multiple stocks
        - Side-by-side analysis
        - Performance metrics
        """)
        if st.button("Compare Stocks", key="compare"):
            st.session_state.page = "Stock Comparison"
            st.rerun()

        st.markdown("### ⚡ Backtesting")
        st.write("""
        - Test investment strategies
        - Risk analysis
        - Performance tracking
        """)
        if st.button("Backtest Strategies", key="backtest"):
            st.session_state.page = "Backtesting"
            st.rerun()

if __name__ == "__main__":
    main()