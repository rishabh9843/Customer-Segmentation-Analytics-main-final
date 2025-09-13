# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import base64

# <<< THIS IS THE CORRECTED IMPORT SECTION
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture # Correctly imported from sklearn.mixture
import hdbscan
# >>> END OF CORRECTION

# Advanced ML & Visualization
import umap.umap_ as umap
import plotly.express as px
import lightgbm as lgb


# --- Page Configuration & Premium Styling ---
st.set_page_config(
    page_title="SegmentIQ v6.1 | Ultimate Edition",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STABLE IMAGE URLS - Updated with working CDN links ---
EDA_ICON_URL = "https://cdn-icons-png.flaticon.com/512/1998/1998557.png"
SEGMENT_ICON_URL = "https://cdn-icons-png.flaticon.com/512/8956/8956264.png"
SIMULATOR_ICON_URL = "https://cdn-icons-png.flaticon.com/512/5261/5261273.png"


# --- PREMIUM UI STYLING (CSS) - NEXUS AI INSPIRED ---
# Replace the existing CSS in your load_css() function with this enhanced version:

# Replace the existing CSS in your load_css() function with this enhanced version:

# Replace the existing CSS in your load_css() function with this enhanced version:

# Replace your entire CSS section with this minimal approach:

# Replace your entire CSS section with this minimal approach:

# Complete working solution - replace your CSS function with this:

# Replace the existing CSS in your load_css() function with this enhanced version:

# Complete solution that works with any Streamlit version:

st.set_page_config(
    page_title="SegmentIQ v6.1 | Ultimate Edition",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Complete solution that works with any Streamlit version (fixed)

import streamlit as st
import streamlit.components.v1 as components

# Page config MUST be first
st.set_page_config(
    page_title="SegmentIQ v6.1 | Ultimate Edition",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
            
            /* UNIVERSAL SIDEBAR FIX - Works with all Streamlit versions */
            
            /* Target ALL possible sidebar containers */
            section[data-testid="stSidebar"],
            .css-1d391kg,
            .css-1lcbmhc, 
            .css-1outpf7,
            .css-1y4p8pa,
            .st-emotion-cache-1cypcdb,
            .st-emotion-cache-6qob1r {
                display: block !important;
                visibility: visible !important;
            }
            
            /* Target ALL possible toggle buttons */
            button[kind="header"],
            div[data-testid="collapsedControl"],
            .css-1rs6os,
            .css-1kyxreq,
            .st-emotion-cache-1rs6os,
            .st-emotion-cache-1kyxreq,
            [data-testid="baseButton-header"] {
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
                z-index: 999999 !important;
                background: linear-gradient(45deg, #00d4ff, #7b68ee) !important;
                border: 2px solid rgba(0, 212, 255, 0.8) !important;
                border-radius: 8px !important;
                color: white !important;
                padding: 8px !important;
                box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                position: relative !important;
            }
            
            /* Hover effects for toggle buttons */
            button[kind="header"]:hover,
            div[data-testid="collapsedControl"]:hover,
            .css-1rs6os:hover,
            .css-1kyxreq:hover,
            .st-emotion-cache-1rs6os:hover,
            .st-emotion-cache-1kyxreq:hover,
            [data-testid="baseButton-header"]:hover {
                background: linear-gradient(45deg, #00bfff, #9370db) !important;
                border-color: rgba(0, 212, 255, 1) !important;
                box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6) !important;
                transform: scale(1.05) !important;
            }
            
            /* Dark theme */
            .main, .stApp, [data-testid="stAppViewContainer"] {
                background: #0a0a0a !important;
                color: #ffffff !important;
                font-family: 'Inter', sans-serif !important;
            }
            
            /* Hide Streamlit branding (but DO NOT hide header itself) */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* header {visibility: hidden;} <-- REMOVED to keep toggle visible */
            .stDeployButton {visibility: hidden;}
            
            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
                border-right: 3px solid rgba(0, 212, 255, 0.6) !important;
                box-shadow: 5px 0 25px rgba(0, 212, 255, 0.3) !important;
            }
            
            section[data-testid="stSidebar"] > div {
                background: transparent !important;
                padding: 1rem !important;
            }
            
            /* Sidebar text styling */
            section[data-testid="stSidebar"] .stMarkdown {
                color: #ffffff !important;
                font-weight: 500 !important;
            }
            
            section[data-testid="stSidebar"] label {
                color: #ffffff !important;
                font-weight: 600 !important;
                text-shadow: 0 0 5px rgba(0, 212, 255, 0.3) !important;
            }
            
            /* Sidebar form elements */
            section[data-testid="stSidebar"] .stSelectbox > div > div {
                background: rgba(0, 212, 255, 0.15) !important;
                border: 2px solid rgba(0, 212, 255, 0.4) !important;
                color: #ffffff !important;
                border-radius: 8px !important;
            }
            
            section[data-testid="stSidebar"] .stFileUploader {
                background: rgba(0, 212, 255, 0.1) !important;
                border: 2px dashed rgba(0, 212, 255, 0.6) !important;
                border-radius: 10px !important;
                padding: 1rem !important;
            }
            
            section[data-testid="stSidebar"] .stButton > button {
                background: linear-gradient(45deg, #00d4ff, #7b68ee) !important;
                border: 2px solid rgba(0, 212, 255, 0.7) !important;
                color: white !important;
                font-weight: 600 !important;
                border-radius: 20px !important;
                padding: 12px 20px !important;
                box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
                transition: all 0.3s ease !important;
                width: 100% !important;
            }
            
            section[data-testid="stSidebar"] .stButton > button:hover {
                background: linear-gradient(45deg, #00bfff, #9370db) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5) !important;
            }
            
            /* Animated background */
            .main::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: 
                    radial-gradient(circle at 20% 50%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 50%, rgba(123, 104, 238, 0.05) 0%, transparent 50%),
                    linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
                pointer-events: none;
                z-index: -1;
                animation: backgroundShift 8s ease-in-out infinite alternate;
            }
            
            @keyframes backgroundShift {
                0% { transform: translateX(-10px); }
                100% { transform: translateX(10px); }
            }
            
            /* Title styling */
            .neon-title {
                font-family: 'Inter', sans-serif;
                font-weight: 900;
                font-size: 4.5rem;
                color: #ffffff;
                text-align: center;
                background: linear-gradient(45deg, #00d4ff, #7b68ee, #ff6b9d, #00d4ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-size: 300% 300%;
                animation: gradientShift 4s ease-in-out infinite;
                margin-bottom: 1rem;
                text-shadow: 
                    0 0 10px rgba(0, 212, 255, 0.8),
                    0 0 20px rgba(0, 212, 255, 0.6),
                    0 0 30px rgba(0, 212, 255, 0.4);
            }
            
            @keyframes gradientShift {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            .neon-title-sidebar {
                font-family: 'Inter', sans-serif;
                font-weight: 800;
                font-size: 1.8rem;
                color: #ffffff;
                text-align: center;
                background: linear-gradient(45deg, #00d4ff, #7b68ee);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
                text-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
            }
            
            /* Main content buttons */
            .stButton > button {
                color: #ffffff !important;
                background: linear-gradient(45deg, #00d4ff, #7b68ee) !important;
                border: 2px solid rgba(0, 212, 255, 0.5) !important;
                border-radius: 25px;
                padding: 15px 30px !important;
                font-weight: 600 !important;
                font-family: 'Inter', sans-serif !important;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3) !important;
                width: 100% !important;
                min-height: 50px !important;
            }
            
            .stButton > button:hover {
                transform: translateY(-3px) !important;
                box-shadow: 0 15px 40px rgba(0, 212, 255, 0.5) !important;
                background: linear-gradient(45deg, #00bfff, #9370db) !important;
            }
            
            /* Custom cards */
            .custom-card {
                background: rgba(26, 26, 46, 0.9) !important;
                border: 2px solid rgba(0, 212, 255, 0.4);
                border-radius: 20px;
                padding: 2.5rem;
                backdrop-filter: blur(20px);
                transition: all 0.4s ease;
                cursor: pointer;
                height: 400px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                position: relative;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                margin-bottom: 2rem;
            }
            
            .custom-card:hover {
                transform: translateY(-15px);
                border-color: rgba(0, 212, 255, 0.8);
                box-shadow: 
                    0 25px 60px rgba(0, 212, 255, 0.3),
                    0 0 40px rgba(0, 212, 255, 0.2);
                background: rgba(26, 26, 46, 0.95) !important;
            }
            
            .custom-card img {
                width: 96px !important;
                height: 96px !important;
                margin-bottom: 1.5rem;
                opacity: 0.9;
                transition: all 0.3s ease;
                display: block !important;
                object-fit: contain;
                filter: brightness(1.2) contrast(1.1);
            }
            
            .custom-card:hover img {
                opacity: 1;
                transform: scale(1.15);
                filter: brightness(1.4) contrast(1.2) saturate(1.2);
            }
            
            .custom-card h3 {
                color: #ffffff !important;
                font-weight: 700;
                font-size: 1.5rem;
                margin-bottom: 1rem;
                font-family: 'Inter', sans-serif;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            }
            
            .custom-card p {
                color: rgba(255, 255, 255, 0.85) !important;
                font-size: 1rem;
                line-height: 1.6;
                font-weight: 400;
            }
            
            /* General styling */
            .subtitle {
                text-align: center;
                color: rgba(255, 255, 255, 0.8) !important;
                font-size: 1.3rem;
                margin-bottom: 3rem;
                font-weight: 400;
                text-shadow: 0 0 5px rgba(0, 212, 255, 0.3);
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
                font-family: 'Inter', sans-serif !important;
            }
            
            p, span, div {
                color: #ffffff;
            }
        </style>
    """, unsafe_allow_html=True)

# Enhanced force sidebar visibility with continuous monitoring (JS injected via components.html)
def force_sidebar_visibility():
    js = """
    <script>
    function ensureToggleButton() {
        const selectors = [
            'button[kind="header"]',
            '[data-testid="collapsedControl"]', 
            '[data-testid="baseButton-header"]',
            '.css-1rs6os',
            '.css-1kyxreq',
            '.st-emotion-cache-1rs6os',
            '.st-emotion-cache-1kyxreq'
        ];
        
        let toggleBtn = null;
        for (let selector of selectors) {
            toggleBtn = document.querySelector(selector);
            if (toggleBtn) break;
        }
        
        if (toggleBtn) {
            toggleBtn.style.cssText = `
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
                position: fixed !important;
                top: 1rem !important;
                left: 1rem !important;
                z-index: 2147483647 !important;
                background: linear-gradient(45deg, #00d4ff, #7b68ee) !important;
                border: 2px solid rgba(0, 212, 255, 0.8) !important;
                border-radius: 8px !important;
                color: white !important;
                padding: 10px !important;
                box-shadow: 0 4px 15px rgba(0, 212, 255, 0.6) !important;
                cursor: pointer !important;
                width: 40px !important;
                height: 40px !important;
                min-width: 40px !important;
                min-height: 40px !important;
            `;
            
            toggleBtn.addEventListener('mouseenter', function() {
                this.style.background = 'linear-gradient(45deg, #00bfff, #9370db)';
                this.style.transform = 'scale(1.1)';
                this.style.boxShadow = '0 8px 25px rgba(0, 212, 255, 0.8)';
            });
            
            toggleBtn.addEventListener('mouseleave', function() {
                this.style.background = 'linear-gradient(45deg, #00d4ff, #7b68ee)';
                this.style.transform = 'scale(1)';
                this.style.boxShadow = '0 4px 15px rgba(0, 212, 255, 0.6)';
            });
        }
        
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.zIndex = '999998';
        }
    }
    
    ensureToggleButton();
    const observer = new MutationObserver(function() {
        ensureToggleButton();
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });
    setInterval(ensureToggleButton, 1000);
    window.addEventListener('load', ensureToggleButton);
    </script>
    """
    # Use components.html to ensure the script actually runs
    components.html(js, height=0)

# Apply the fixes
load_css()
force_sidebar_visibility()


# --- Session State Initialization ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'results' not in st.session_state: st.session_state.results = {}

# --- All backend data processing functions (cached for performance) ---
@st.cache_data
def load_data(uploaded_file): return pd.read_csv(uploaded_file, encoding='latin1')

@st.cache_data
def preprocess_data(df_raw):
    df = df_raw.dropna(subset=['CustomerID', 'InvoiceDate']).copy()
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]; df['Sales'] = df['Quantity'] * df['UnitPrice']
    Q1, Q3 = df['Sales'].quantile(0.25), df['Sales'].quantile(0.75)
    df = df[~((df['Sales'] < (Q1 - 1.5 * (Q3 - Q1))) | (df['Sales'] > (Q3 + 1.5 * (Q3 - Q1))))]
    return df

@st.cache_data
def engineer_features(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    ref_date = df['InvoiceDate'].max() + timedelta(days=1)
    features = df.groupby('CustomerID').agg(
        recency_days=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary_value=('Sales', 'sum'),
    ).reset_index(); features.set_index('CustomerID', inplace=True)
    return features.fillna(0)

@st.cache_resource
def run_clustering(features_df, algorithm, n_clusters, min_cluster_size):
    scaler = RobustScaler(); features_scaled = scaler.fit_transform(features_df)
    if algorithm == 'K-Means': model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'Gaussian Mixture': model = GaussianMixture(n_components=n_clusters, random_state=42)
    else: model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(features_scaled)
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(features_scaled, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features_scaled, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(features_scaled, labels)
    return labels, features_scaled, metrics

@st.cache_resource
def train_predictive_models(_features_df, _labels):
    df = _features_df.copy(); df['cluster'] = _labels
    df['is_churn'] = (df['recency_days'] > df['recency_days'].quantile(0.75)).astype(int)
    X_churn, y_churn = df.drop(columns=['is_churn', 'cluster']), df['is_churn']
    churn_model = lgb.LGBMClassifier(random_state=42).fit(X_churn, y_churn)
    predictions_df = _features_df.copy(); predictions_df['cluster'] = _labels
    predictions_df['churn_probability'] = churn_model.predict_proba(X_churn)[:, 1]
    feature_importance = pd.DataFrame({'feature': X_churn.columns, 'importance': churn_model.feature_importances_}).sort_values('importance', ascending=False)
    return predictions_df, churn_model, feature_importance

@st.cache_data
def generate_personas(df):
    personas = {}
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1: continue
        segment_data = df[df['cluster'] == cluster_id]
        r, f, m = segment_data['recency_days'].mean(), segment_data['frequency'].mean(), segment_data['monetary_value'].mean()
        if r < 90 and f > 10 and m > 2000: persona = "🏆 VIP Champions"
        elif r > 180 and f < 2: persona = "💤 At-Risk / Dormant"
        else: persona = f"🌿 Promising Segment {cluster_id}"
        personas[cluster_id] = {'persona': persona, 'size': len(segment_data), 'avg_recency': r, 'avg_frequency': f, 'avg_monetary': m}
    return personas

# --- UI RENDERING FUNCTIONS ---
st.markdown(
    """
    <style>
    /* Ensure sidebar is visible */
    section[data-testid="stSidebar"] {
        display: block !important;
    }

    /* Ensure the toggle (arrow) is visible */
    button[kind="header"] {
        display: block !important;
    }

    /* Sometimes the toggle is hidden in newer versions */
    div[data-testid="collapsedControl"] {
        display: block !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def render_sidebar():
    with st.sidebar:
        st.markdown('<h1 class="neon-title-sidebar">SegmentIQ v6.1</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Ultimate AI Analytics Suite</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        uploaded_file = st.file_uploader("📁 Upload Your Sales Data", type="csv", help="Upload a CSV file with your sales data")
        
        st.markdown("### ⚙️ Configuration")
        algorithm = st.selectbox("🤖 Algorithm", ('HDBSCAN', 'K-Means', 'Gaussian Mixture'))
        
        if algorithm in ['K-Means', 'Gaussian Mixture']: 
            param = st.slider("🎯 Number of Segments (K)", 2, 15, 5, 1)
        else: 
            param = st.slider("👥 Minimum Segment Size", 5, 100, 30, 5)
        
        st.markdown("---")
        
        # ENHANCED RUN BUTTON
        st.markdown("### 🚀 Launch Analysis")
        run_button = st.button("🚀 Run Full Analysis", help="Start the complete AI analysis", type="primary", use_container_width=True)
        
        if uploaded_file:
            st.success("✅ Data file loaded successfully!")
        
        return uploaded_file, algorithm, param, run_button

def render_homepage():
    st.markdown('<h1 class="neon-title">SegmentIQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">The ultimate AI-powered customer analytics platform for strategic intelligence</p>', unsafe_allow_html=True)
    
    # Create centered layout for cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown(f'<div class="custom-card"> <img src="{EDA_ICON_URL}"> <h3>Exploratory Analysis</h3> <p>Deep dive into your dataset with advanced statistical profiling and insights.</p> </div>', unsafe_allow_html=True)
            if st.button("Go to EDA", key="eda", use_container_width=True):
                st.session_state.page = 'eda'
                st.rerun()

    with col2:
        with st.container():
            st.markdown(f'<div class="custom-card"> <img src="{SEGMENT_ICON_URL}"> <h3>AI Segmentation</h3> <p>Discover customer personas using machine learning and behavioral analytics.</p> </div>', unsafe_allow_html=True)
            if st.button("Go to Segmentation", key="analysis", use_container_width=True):
                st.session_state.page = 'analysis'
                st.rerun()
                
    with col3:
        with st.container():
            st.markdown(f'<div class="custom-card"> <img src="{SIMULATOR_ICON_URL}"> <h3>Churn Simulator</h3> <p>Predict customer behavior with interactive what-if scenario modeling.</p> </div>', unsafe_allow_html=True)
            if st.button("Go to Simulator", key="simulator", use_container_width=True):
                st.session_state.page = 'simulator'
                st.rerun()


def render_eda_page(df):
    st.markdown('<h1 class="neon-title">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("⬅️ Home"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        st.markdown('<p class="subtitle">Comprehensive data profiling and statistical insights</p>', unsafe_allow_html=True)
    
    # Data preview
    st.markdown("### 🔍 Data Overview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("👥 Unique Customers", f"{df['CustomerID'].nunique():,}")
    with col3:
        st.metric("💰 Total Revenue", f"${df['Sales'].sum():,.2f}")
    with col4:
        st.metric("📈 Avg Order Value", f"${df['Sales'].mean():.2f}")
    
    # Visualizations - IMPROVED VISIBILITY
    st.markdown("### 📈 Sales Distribution Analysis")
    fig = px.histogram(
        df, 
        x="Sales", 
        nbins=50, 
        title="Distribution of Transaction Values",
        color_discrete_sequence=['#00d4ff']
    )
    fig.update_layout(
        paper_bgcolor='rgba(10,10,10,0.9)', 
        plot_bgcolor='rgba(26,26,46,0.8)', 
        font_color='#ffffff',
        title_font_color='#00d4ff',
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff')
    )
    st.plotly_chart(fig, use_container_width=True)

def render_analysis_dashboard(results):
    st.markdown('<h1 class="neon-title">🧠 AI Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("⬅️ Home"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        st.markdown('<p class="subtitle">Advanced customer segmentation and behavioral insights</p>', unsafe_allow_html=True)
    
    personas_df = pd.DataFrame.from_dict(results['personas'], orient='index')
    
    tabs = st.tabs(["🏆 Customer Personas", "📊 Segment Analytics", "🧠 AI Insights", "🔮 Predictions"])
    
    with tabs[0]:
        st.markdown("### 👑 Discovered Customer Personas")
        
        # Download report button
        report = "# Customer Personas Report\n\n"
        for cid, data in results['personas'].items(): 
            report += f"## {data['persona']} (Segment {cid})\n"
            report += f"- Customer Count: {data['size']:,}\n"
            report += f"- Average Recency: {data['avg_recency']:.0f} days\n"
            report += f"- Average Frequency: {data['avg_frequency']:.1f} transactions\n"
            report += f"- Average Monetary Value: ${data['avg_monetary']:,.2f}\n\n"
        
        st.download_button(
            "📥 Download Personas Report", 
            report, 
            "customer_personas_report.md",
            mime="text/markdown"
        )
        
        # Display personas
        for cid, data in results['personas'].items():
            with st.expander(f"**{data['persona']}** - Segment {cid}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("👥 Customer Count", f"{data['size']:,}")
                with col2:
                    st.metric("⏱️ Avg. Recency", f"{data['avg_recency']:.0f} days")
                with col3:
                    st.metric("🔄 Avg. Frequency", f"{data['avg_frequency']:.1f}")
                with col4:
                    st.metric("💰 Avg. Spend", f"${data['avg_monetary']:,.2f}")
    
    with tabs[1]:
        sub_tabs = st.tabs(["🎯 Performance Matrix", "📈 Behavior Patterns", "⚖️ Segment Comparison"])
        
        with sub_tabs[0]:
            st.markdown("### 🎯 Segment Performance Matrix")
            fig = px.scatter(
                personas_df, 
                x="avg_recency", 
                y="avg_monetary", 
                size="size", 
                color="persona", 
                hover_name="persona", 
                size_max=60, 
                title="Customer Segment Performance Overview",
                color_discrete_sequence=['#00d4ff', '#7b68ee', '#ff6b9d', '#00ff9f', '#ff9f40', '#ff4081']
            )
            fig.update_layout(
                paper_bgcolor='rgba(10,10,10,0.9)', 
                plot_bgcolor='rgba(26,26,46,0.8)', 
                font_color='#ffffff',
                title_font_color='#00d4ff',
                xaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff')
            )
            # Increase marker size and add border for better visibility
            fig.update_traces(marker=dict(line=dict(width=2, color='rgba(255,255,255,0.8)')))
            st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[1]:
            st.markdown("### 📈 Customer Behavior Patterns")
            behavior_df = results['predictions_df'].copy()
            behavior_df['persona'] = [results['personas'].get(l, {}).get('persona', 'Outlier') for l in behavior_df['cluster']]
            
            fig = px.scatter(
                behavior_df, 
                x='frequency', 
                y='monetary_value', 
                color='persona', 
                title="Purchase Frequency vs Monetary Value",
                color_discrete_sequence=['#00d4ff', '#7b68ee', '#ff6b9d', '#00ff9f', '#ff9f40', '#ff4081']
            )
            fig.update_layout(
                paper_bgcolor='rgba(10,10,10,0.9)', 
                plot_bgcolor='rgba(26,26,46,0.8)', 
                font_color='#ffffff',
                title_font_color='#00d4ff',
                xaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff')
            )
            # Increase marker size and opacity for better visibility
            fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.5)')))
            st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[2]:
            st.markdown("### ⚖️ Segment Comparison Tool")
            col1, col2 = st.columns(2)
            seg_list = personas_df['persona'].tolist()
            
            with col1:
                seg_a = st.selectbox("Select Segment A", seg_list, index=0)
            with col2:
                seg_b = st.selectbox("Select Segment B", seg_list, index=1 if len(seg_list) > 1 else 0)
            
            comparison_df = pd.DataFrame([
                personas_df[personas_df['persona'] == seg_a].iloc[0], 
                personas_df[personas_df['persona'] == seg_b].iloc[0]
            ])
            st.dataframe(comparison_df, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 🧠 AI Model Insights")
        sub_tabs = st.tabs(["🎯 Feature Importance", "🌐 3D Visualization", "📊 Model Metrics"])
        
        with sub_tabs[0]:
            st.markdown("#### 🎯 Churn Prediction Feature Importance")
            fig = px.bar(
                results['feature_importance'], 
                x='importance', 
                y='feature', 
                orientation='h', 
                title='Feature Impact on Churn Prediction',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor='rgba(10,10,10,0.9)', 
                plot_bgcolor='rgba(26,26,46,0.8)', 
                font_color='#ffffff',
                title_font_color='#00d4ff',
                xaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)', color='#ffffff')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[1]:
            st.markdown("#### 🌐 Interactive 3D Customer Segments")
            embedding = umap.UMAP(n_components=3, random_state=42).fit_transform(results['features_scaled'])
            viz_df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
            viz_df['Persona'] = [results['personas'].get(l, {}).get('persona', 'Outlier') for l in results['labels']]
            
            fig = px.scatter_3d(
                viz_df, 
                x='x', 
                y='y', 
                z='z', 
                color='Persona', 
                title="3D Customer Segment Visualization",
                color_discrete_sequence=['#00d4ff', '#7b68ee', '#ff6b9d', '#00ff9f', '#ff9f40', '#ff4081']
            )
            # Enhanced plot styling for dark theme with better visibility
            fig.update_layout(
                scene=dict(
                    bgcolor='rgba(26,26,46,0.9)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.3)', tickcolor='#ffffff', color='#ffffff'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.3)', tickcolor='#ffffff', color='#ffffff'),
                    zaxis=dict(gridcolor='rgba(255,255,255,0.3)', tickcolor='#ffffff', color='#ffffff')
                ),
                paper_bgcolor='rgba(10,10,10,0.9)',
                font_color='#ffffff',
                title_font_color='#00d4ff',
                title_font_size=16,
                title_font_family='Inter'
            )
            # Increase marker size and add borders for better visibility
            fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.5)')))
            st.plotly_chart(fig, use_container_width=True)
        
        with sub_tabs[2]:
            st.markdown("#### 📊 Clustering Quality Metrics")
            metrics = results.get('metrics', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{metrics.get('silhouette', 0):.3f}", help="Higher is better (max: 1)")
            with col2:
                st.metric("Calinski-Harabasz", f"{metrics.get('calinski_harabasz', 0):,.0f}", help="Higher indicates better clustering")
            with col3:
                st.metric("Davies-Bouldin", f"{metrics.get('davies_bouldin', 0):.3f}", help="Lower is better (min: 0)")
    
    with tabs[3]:
        st.markdown("### 🔮 Customer Predictions & Lookup")
        
        df_display = results['predictions_df'].reset_index()
        df_display['Persona'] = [results['personas'].get(l, {}).get('persona', 'Outlier') for l in df_display['cluster']]
        
        # Search functionality
        col1, col2 = st.columns([2, 1])
        with col1:
            search_id = st.text_input("🔍 Search Customer ID", placeholder="Enter customer ID to search...")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🔄 Reset Filter"):
                search_id = ""
        
        if search_id:
            df_display = df_display[df_display['CustomerID'].str.contains(search_id, case=False, na=False)]
        
        # Display results
        display_columns = ['CustomerID', 'Persona', 'churn_probability', 'recency_days', 'frequency', 'monetary_value']
        st.dataframe(
            df_display[display_columns].style.format({
                'churn_probability': '{:.1%}',
                'monetary_value': '${:,.2f}'
            }), 
            use_container_width=True
        )

def render_simulator_page(results, features_df):
    st.markdown('<h1 class="neon-title">🔮 Churn Risk Simulator</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("⬅️ Home"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        st.markdown('<p class="subtitle">Interactive what-if analysis for customer churn prediction</p>', unsafe_allow_html=True)
    
    churn_model = results['churn_model']
    
    # Customer selection
    st.markdown("### 👤 Select Customer")
    customer_list = features_df.index.tolist()
    selected_customer = st.selectbox("Choose a customer to analyze", customer_list, help="Select any customer from your dataset")
    
    if selected_customer:
        customer_data = features_df.loc[selected_customer]
        
        # Current customer info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Recency", f"{int(customer_data['recency_days'])} days")
        with col2:
            st.metric("Current Frequency", f"{int(customer_data['frequency'])}")
        with col3:
            st.metric("Current Monetary", f"${customer_data['monetary_value']:,.2f}")
        with col4:
            original_churn_prob = customer_data.get('churn_probability', 0)
            st.metric("Current Churn Risk", f"{original_churn_prob:.1%}")
        
        st.markdown("---")
        
        # Simulation controls
        st.markdown("### ⚙️ Adjust Customer Behavior")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_recency = st.slider(
                "📅 Recency (Days since last purchase)", 
                0, 365, 
                int(customer_data['recency_days']),
                help="Lower values indicate more recent purchases"
            )
        
        with col2:
            new_frequency = st.slider(
                "🔄 Frequency (Number of purchases)", 
                1, 100, 
                max(1, int(customer_data['frequency'])),
                help="Higher values indicate more loyal customers"
            )
        
        with col3:
            new_monetary = st.slider(
                "💰 Monetary Value ($)", 
                0, 10000, 
                int(customer_data['monetary_value']),
                help="Total amount spent by the customer"
            )
        
        # Prediction
        simulated_data = pd.DataFrame([{
            'recency_days': new_recency,
            'frequency': new_frequency,
            'monetary_value': new_monetary
        }])
        
        churn_prob = churn_model.predict_proba(simulated_data)[0][1]
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        # Results display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Churn Risk Level")
            st.progress(churn_prob)
            
            # Risk categorization
            if churn_prob < 0.3:
                risk_level = "🟢 Low Risk"
                risk_color = "success"
            elif churn_prob < 0.6:
                risk_level = "🟡 Medium Risk"
                risk_color = "warning"
            else:
                risk_level = "🔴 High Risk"
                risk_color = "error"
            
            st.markdown(f"**Risk Level:** {risk_level}")
        
        with col2:
            delta_value = churn_prob - original_churn_prob
            st.metric(
                "Updated Churn Probability", 
                f"{churn_prob:.1%}", 
                delta=f"{delta_value:.1%}",
                delta_color="inverse"
            )
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            if churn_prob > 0.6:
                st.warning("⚠️ **High Risk Customer** - Immediate intervention recommended")
                st.markdown("- Offer personalized discount")
                st.markdown("- Reach out with customer service call")
                st.markdown("- Send targeted retention campaign")
            elif churn_prob > 0.3:
                st.info("ℹ️ **Medium Risk Customer** - Monitor and engage")
                st.markdown("- Send engagement email campaign")
                st.markdown("- Offer loyalty program benefits")
                st.markdown("- Monitor purchase behavior closely")
            else:
                st.success("✅ **Low Risk Customer** - Maintain engagement")
                st.markdown("- Continue regular marketing")
                st.markdown("- Consider upselling opportunities")
                st.markdown("- Reward loyalty with exclusive offers")

# --- Main App Logic ---
if __name__ == "__main__":
    uploaded_file, algorithm, param, run_button = render_sidebar()

    # Handle analysis trigger
    if run_button and uploaded_file:
        st.session_state.analysis_run = True
        st.session_state.page = 'analysis'
        st.rerun()

    # Page routing with enhanced styling
    if st.session_state.page == 'home':
        render_homepage()
        
    elif st.session_state.page == 'eda':
        if uploaded_file:
            with st.spinner("🔍 Processing data..."):
                processed_data = preprocess_data(load_data(uploaded_file))
            render_eda_page(processed_data)
        else:
            st.warning("⚠️ Please upload a CSV file to perform Exploratory Data Analysis.")
            if st.button("🏠 Return to Home"):
                st.session_state.page = 'home'
                st.rerun()
                
    elif st.session_state.page in ['analysis', 'simulator']:
        if not st.session_state.analysis_run or not uploaded_file:
            st.warning("⚠️ Please upload data and run a full analysis from the sidebar first.")
            if st.button("🏠 Return to Home"):
                st.session_state.page = 'home'
                st.rerun()
        else:
            # Run analysis if not already done
            if 'personas' not in st.session_state.results:
                with st.spinner("🚀 Running advanced AI analysis... This may take a moment."):
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Processing data...")
                    progress_bar.progress(20)
                    features_df = engineer_features(preprocess_data(load_data(uploaded_file)))
                    
                    status_text.text("🤖 Running clustering algorithms...")
                    progress_bar.progress(40)
                    n_clusters = param if algorithm in ['K-Means', 'Gaussian Mixture'] else 5
                    min_size = param if algorithm == 'HDBSCAN' else 30
                    labels, scaled_features, metrics = run_clustering(features_df, algorithm, n_clusters, min_size)
                    
                    status_text.text("🧠 Training predictive models...")
                    progress_bar.progress(60)
                    predictions, churn_model, feat_imp = train_predictive_models(features_df, labels)
                    
                    status_text.text("👑 Generating customer personas...")
                    progress_bar.progress(80)
                    personas = generate_personas(predictions)
                    
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)
                    
                    # Store results
                    st.session_state.results = {
                        'personas': personas,
                        'labels': labels,
                        'features_scaled': scaled_features,
                        'predictions_df': predictions,
                        'metrics': metrics,
                        'churn_model': churn_model,
                        'feature_importance': feat_imp
                    }
                    
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()
            
            # Render appropriate page
            if st.session_state.page == 'analysis':
                render_analysis_dashboard(st.session_state.results)
            elif st.session_state.page == 'simulator':
                render_simulator_page(st.session_state.results, st.session_state.results['predictions_df'])









