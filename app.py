# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import base64

# Machine Learning Imports
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import hdbscan

# Advanced ML & Visualization
import umap.umap_ as umap
import plotly.express as px
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Configuration & Premium Styling ---
st.set_page_config(
    page_title="SegmentIQ v6.1 | Ultimate Edition",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STABLE IMAGE URLS ---
EDA_ICON_URL = "https://cdn-icons-png.flaticon.com/512/1998/1998557.png"
SEGMENT_ICON_URL = "https://cdn-icons-png.flaticon.com/512/8956/8956264.png"
SIMULATOR_ICON_URL = "https://cdn-icons-png.flaticon.com/512/5261/5261273.png"

# CSS and JavaScript functions
def load_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
            
            /* UNIVERSAL SIDEBAR FIX - Works with all Streamlit versions */
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
            
            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
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

def force_sidebar_visibility():
    import streamlit.components.v1 as components
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
        }
        
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.zIndex = '999998';
        }
    }
    
    ensureToggleButton();
    const observer = new MutationObserver(ensureToggleButton);
    observer.observe(document.body, { childList: true, subtree: true });
    setInterval(ensureToggleButton, 1000);
    </script>
    """
    components.html(js, height=0)

# Apply styling
load_css()
force_sidebar_visibility()

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# --- Data Processing Functions ---
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_data(df_raw):
    """Clean and preprocess the raw data"""
    try:
        df = df_raw.copy()
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
        
        # Convert data types
        df['CustomerID'] = df['CustomerID'].astype(str)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        
        # Filter positive values
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        # Calculate sales
        df['Sales'] = df['Quantity'] * df['UnitPrice']
        
        # Remove outliers using IQR method
        Q1 = df['Sales'].quantile(0.25)
        Q3 = df['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['Sales'] < (Q1 - 1.5 * IQR)) | (df['Sales'] > (Q3 + 1.5 * IQR)))]
        
        final_rows = len(df)
        st.info(f"Data preprocessing: {initial_rows:,} → {final_rows:,} rows ({((final_rows/initial_rows)*100):.1f}% retained)")
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

@st.cache_data
def engineer_features(df):
    """Create RFM features for clustering"""
    try:
        df = df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # Reference date (one day after the latest transaction)
        ref_date = df['InvoiceDate'].max() + timedelta(days=1)
        
        # Create RFM features
        features = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'Sales': 'sum'  # Monetary
        }).reset_index()
        
        features.columns = ['CustomerID', 'recency_days', 'frequency', 'monetary_value']
        features.set_index('CustomerID', inplace=True)
        
        # Handle any remaining NaN values
        features = features.fillna(0)
        
        st.success(f"Features engineered for {len(features):,} unique customers")
        return features
        
    except Exception as e:
        st.error(f"Error engineering features: {str(e)}")
        return None

def run_clustering(features_df, algorithm, n_clusters, min_cluster_size):
    """Run clustering algorithm on the features"""
    try:
        # Scale features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Initialize model based on algorithm
        if algorithm == 'K-Means':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'Gaussian Mixture':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        else:  # HDBSCAN
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        
        # Fit and predict
        labels = model.fit_predict(features_scaled)
        
        # Calculate clustering metrics
        metrics = {}
        unique_labels = np.unique(labels)
        
        if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
            # Only calculate metrics if we have meaningful clusters
            valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
            if np.sum(valid_mask) > 1:
                metrics['silhouette'] = silhouette_score(features_scaled[valid_mask], labels[valid_mask])
                metrics['calinski_harabasz'] = calinski_harabasz_score(features_scaled[valid_mask], labels[valid_mask])
                metrics['davies_bouldin'] = davies_bouldin_score(features_scaled[valid_mask], labels[valid_mask])
        
        n_clusters_found = len(unique_labels)
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        
        st.success(f"Clustering complete: {n_clusters_found} clusters found, {n_noise} noise points")
        
        return labels, features_scaled, metrics, scaler
        
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        return None, None, {}, None

def train_predictive_models(features_df, labels):
    """Train churn prediction model"""
    try:
        df = features_df.copy()
        df['cluster'] = labels
        
        # Define churn based on recency (customers who haven't purchased in the top 25% of days)
        churn_threshold = df['recency_days'].quantile(0.75)
        df['is_churn'] = (df['recency_days'] > churn_threshold).astype(int)
        
        # Prepare data for modeling
        X = df[['recency_days', 'frequency', 'monetary_value']]
        y = df['is_churn']
        
        # Train LightGBM model
        model = lgb.LGBMClassifier(
            random_state=42,
            verbosity=-1,
            force_col_wise=True
        )
        model.fit(X, y)
        
        # Make predictions
        churn_proba = model.predict_proba(X)[:, 1]
        
        # Create results dataframe
        predictions_df = features_df.copy()
        predictions_df['cluster'] = labels
        predictions_df['churn_probability'] = churn_proba
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.success(f"Predictive model trained with {len(X):,} samples")
        
        return predictions_df, model, feature_importance
        
    except Exception as e:
        st.error(f"Error training predictive models: {str(e)}")
        return None, None, None

def generate_personas(predictions_df):
    """Generate customer personas based on clustering results"""
    try:
        personas = {}
        
        for cluster_id in sorted(predictions_df['cluster'].unique()):
            if cluster_id == -1:  # Skip noise cluster
                continue
                
            segment_data = predictions_df[predictions_df['cluster'] == cluster_id]
            
            # Calculate segment statistics
            avg_recency = segment_data['recency_days'].mean()
            avg_frequency = segment_data['frequency'].mean()
            avg_monetary = segment_data['monetary_value'].mean()
            segment_size = len(segment_data)
            
            # Assign persona based on RFM characteristics
            if avg_recency <= 30 and avg_frequency >= 10 and avg_monetary >= 1000:
                persona = "🏆 VIP Champions"
            elif avg_recency <= 60 and avg_frequency >= 5:
                persona = "💎 Loyal Customers"
            elif avg_recency > 180:
                persona = "💤 At-Risk/Dormant"
            elif avg_frequency >= 8:
                persona = "🔄 Frequent Buyers"
            elif avg_monetary >= 500:
                persona = "💰 Big Spenders"
            else:
                persona = f"🌿 Potential Segment"
            
            personas[cluster_id] = {
                'persona': persona,
                'size': segment_size,
                'avg_recency': avg_recency,
                'avg_frequency': avg_frequency,
                'avg_monetary': avg_monetary
            }
        
        st.success(f"Generated {len(personas)} customer personas")
        return personas
        
    except Exception as e:
        st.error(f"Error generating personas: {str(e)}")
        return {}

# --- UI RENDERING FUNCTIONS ---
def render_sidebar():
    with st.sidebar:
        st.markdown('<h1 class="neon-title-sidebar">SegmentIQ v6.1</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Ultimate AI Analytics Suite</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📁 Upload Your Sales Data", 
            type="csv", 
            help="Upload a CSV file with your sales data (should contain CustomerID, InvoiceDate, Quantity, UnitPrice columns)"
        )
        
        st.markdown("### ⚙️ Configuration")
        algorithm = st.selectbox("🤖 Algorithm", ('K-Means', 'Gaussian Mixture', 'HDBSCAN'))
        
        if algorithm in ['K-Means', 'Gaussian Mixture']: 
            param = st.slider("🎯 Number of Segments (K)", 2, 15, 5, 1)
        else: 
            param = st.slider("👥 Minimum Segment Size", 5, 100, 30, 5)
        
        st.markdown("---")
        
        # Run button
        st.markdown("### 🚀 Launch Analysis")
        run_button = st.button(
            "🚀 Run Full Analysis", 
            help="Start the complete AI analysis", 
            type="primary", 
            use_container_width=True
        )
        
        if uploaded_file:
            st.success("✅ Data file loaded successfully!")
        
        return uploaded_file, algorithm, param, run_button

def render_homepage():
    st.markdown('<h1 class="neon-title">SegmentIQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">The ultimate AI-powered customer analytics platform for strategic intelligence</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="custom-card"><img src="{EDA_ICON_URL}"><h3>Exploratory Analysis</h3><p>Deep dive into your dataset with advanced statistical profiling and insights.</p></div>', unsafe_allow_html=True)
        if st.button("Go to EDA", key="eda", use_container_width=True):
            st.session_state.page = 'eda'
            st.rerun()

    with col2:
        st.markdown(f'<div class="custom-card"><img src="{SEGMENT_ICON_URL}"><h3>AI Segmentation</h3><p>Discover customer personas using machine learning and behavioral analytics.</p></div>', unsafe_allow_html=True)
        if st.button("Go to Segmentation", key="analysis", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()
                
    with col3:
        st.markdown(f'<div class="custom-card"><img src="{SIMULATOR_ICON_URL}"><h3>Churn Simulator</h3><p>Predict customer behavior with interactive what-if scenario modeling.</p></div>', unsafe_allow_html=True)
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
    
    # Visualizations
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
            st.markdown("<br>", unsafe_allow_html=True)
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
            elif churn_prob < 0.6:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🔴 High Risk"
            
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

# --- Main Application Logic ---
def main():
    uploaded_file, algorithm, param, run_button = render_sidebar()

    # Handle analysis trigger
    if run_button and uploaded_file:
        # Clear previous results and start fresh analysis
        st.session_state.results = {}
        st.session_state.analysis_run = True
        st.session_state.page = 'analysis'
        st.session_state.uploaded_data = uploaded_file
        st.rerun()

    # Page routing
    if st.session_state.page == 'home':
        render_homepage()
        
    elif st.session_state.page == 'eda':
        if uploaded_file:
            with st.spinner("🔍 Processing data..."):
                raw_data = load_data(uploaded_file)
                if raw_data is not None:
                    processed_data = preprocess_data(raw_data)
                    if processed_data is not None:
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
                    
                    try:
                        status_text.text("📊 Loading and processing data...")
                        progress_bar.progress(10)
                        
                        raw_data = load_data(uploaded_file)
                        if raw_data is None:
                            st.error("Failed to load data. Please check your file format.")
                            return
                        
                        processed_data = preprocess_data(raw_data)
                        if processed_data is None:
                            st.error("Failed to preprocess data. Please check your data columns.")
                            return
                        
                        status_text.text("🔧 Engineering features...")
                        progress_bar.progress(30)
                        
                        features_df = engineer_features(processed_data)
                        if features_df is None:
                            st.error("Failed to engineer features. Please check your data structure.")
                            return
                        
                        status_text.text("🤖 Running clustering algorithms...")
                        progress_bar.progress(50)
                        
                        n_clusters = param if algorithm in ['K-Means', 'Gaussian Mixture'] else 5
                        min_size = param if algorithm == 'HDBSCAN' else 30
                        
                        clustering_results = run_clustering(features_df, algorithm, n_clusters, min_size)
                        if clustering_results[0] is None:
                            st.error("Clustering failed. Please try different parameters.")
                            return
                        
                        labels, scaled_features, metrics, scaler = clustering_results
                        
                        status_text.text("🧠 Training predictive models...")
                        progress_bar.progress(70)
                        
                        model_results = train_predictive_models(features_df, labels)
                        if model_results[0] is None:
                            st.error("Model training failed.")
                            return
                        
                        predictions, churn_model, feat_imp = model_results
                        
                        status_text.text("👑 Generating customer personas...")
                        progress_bar.progress(90)
                        
                        personas = generate_personas(predictions)
                        if not personas:
                            st.error("Failed to generate personas.")
                            return
                        
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
                            'feature_importance': feat_imp,
                            'scaler': scaler
                        }
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        return
                    finally:
                        # Clean up progress indicators
                        progress_bar.empty()
                        status_text.empty()
            
            # Render appropriate page
            if st.session_state.page == 'analysis':
                render_analysis_dashboard(st.session_state.results)
            elif st.session_state.page == 'simulator':
                render_simulator_page(st.session_state.results, st.session_state.results['predictions_df'])

# Run the main application
if __name__ == "__main__":
    main()
