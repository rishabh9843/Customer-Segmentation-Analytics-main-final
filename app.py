import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import hdbscan
import lightgbm as lgb
import umap

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Customer Segmentation Analytics", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); color: #ffffff;}
    h1, h2, h3 {color: #00d4ff !important;}
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #7b68ee);
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        border: none;
        font-weight: 600;
    }
    .metric-card {
        background: rgba(0, 212, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Customer Segmentation Analytics")
st.markdown("**AI-powered customer insights using Sentence Transformers (BERT), Ensemble Clustering, and Predictive Modeling**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type="csv")
    
    algorithm = st.selectbox("Clustering Algorithm", ['K-Means', 'HDBSCAN'])
    
    # K-Means settings
    n_clusters = st.slider("Number of Segments (K)", 2, 10, 5) if algorithm == 'K-Means' else None
    
    use_nlp = st.checkbox("Enable NLP Features", value=True, help="Extract semantic features using Sentence Transformers")
    
    if uploaded_file:
        run_analysis = st.button("🚀 Run Analysis", use_container_width=True)
    else:
        st.info("Upload CSV to begin")

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================

@st.cache_data
def load_and_process(file):
    """Load and clean the raw data"""
    df = pd.read_csv(file)
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    return df

@st.cache_data
def engineer_rfm_features(df):
    """Create Recency, Frequency, Monetary features"""
    ref_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    features = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'Sales': 'sum'  # Monetary
    })
    features.columns = ['recency_days', 'frequency', 'monetary_value']
    return features

@st.cache_data
def extract_nlp_features(df):
    """
    Transformer NLP: Uses a pre-trained Sentence Transformer (BERT-based)
    to convert customer purchase history into semantic vectors.
    """
    if 'Description' not in df.columns:
        return pd.DataFrame()
    
    # 1. Cleaning: Lowercase and handle missing
    df_clean = df.copy()
    df_clean['Description'] = df_clean['Description'].fillna('').astype(str).str.lower()
    
    # 2. Aggregation: Create one long "story" for each customer
    # "bag bag lunch box..."
    customer_group = df_clean.groupby('CustomerID')['Description'].apply(lambda x: ' '.join(x))
    
    # 3. Load the Transformer Model (The "Brain")
    # 'all-MiniLM-L6-v2' is the industry standard for fast/lightweight embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Encoding: Convert text to numbers (384 Dimensions)
    embeddings = model.encode(customer_group.tolist())
    
    # 5. Dimensionality Reduction (PCA)
    # Squash 384 dimensions down to 5 for efficient clustering
    pca = PCA(n_components=5, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 6. Formatting
    nlp_features = pd.DataFrame(reduced_embeddings, index=customer_group.index)
    nlp_features.columns = [f'nlp_dim_{i+1}' for i in range(5)]
    
    # Add Variety Metric
    nlp_features['unique_products_count'] = df_clean.groupby('CustomerID')['StockCode'].nunique()
    
    return nlp_features

def run_clustering(features_df, algo, k):
    """Perform ensemble clustering with adaptive settings"""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    if algo == 'K-Means':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        # Adaptive settings for HDBSCAN
        data_size = features_df.shape[0]
        min_cluster_size = int(max(3, min(50, data_size * 0.015)))
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric='euclidean')
    
    labels = model.fit_predict(X_scaled)
    return labels, X_scaled

def train_churn_model(features_df, labels):
    """Train LightGBM churn prediction model"""
    df = features_df.copy()
    df['cluster'] = labels
    
    # Define churn (high recency = at risk)
    churn_threshold = df['recency_days'].quantile(0.75)
    df['is_churn'] = (df['recency_days'] > churn_threshold).astype(int)
    
    # Use RFM features for prediction
    X = df[['recency_days', 'frequency', 'monetary_value']]
    y = df['is_churn']
    
    # LightGBM model
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1, n_estimators=100)
    model.fit(X, y)
    
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return df, model, importance

def estimate_clv(predictions_df):
    """Estimate Customer Lifetime Value (Heuristic)"""
    predictions_df['estimated_clv'] = (
        predictions_df['monetary_value'] * (1 - predictions_df['churn_probability']) * (predictions_df['frequency'] / 12)
    )
    return predictions_df

def create_personas(predictions_df):
    """Generate GROUP-LEVEL customer personas"""
    personas = {}
    unique_labels = sorted(predictions_df['cluster'].unique())
    
    for cluster_id in unique_labels:
        if cluster_id == -1: continue # Skip noise
        
        segment = predictions_df[predictions_df['cluster'] == cluster_id]
        
        avg_r = segment['recency_days'].mean()
        avg_f = segment['frequency'].mean()
        avg_m = segment['monetary_value'].mean()
        avg_clv = segment['estimated_clv'].mean()
        
        # Rule-based Persona Naming for the GROUP
        if avg_r <= 40 and avg_f >= 5 and avg_m >= 1000:
            persona = "🏆 VIP Champions"
            strategy = "Exclusive benefits, personalized service"
        elif avg_r <= 60 and avg_f >= 3:
            persona = "💎 Loyal Customers"
            strategy = "Upsell premium, referral programs"
        elif avg_r > 150:
            persona = "💤 At-Risk/Dormant"
            strategy = "Win-back campaigns, special offers"
        elif avg_f >= 5:
            persona = "🔄 Frequent Buyers"
            strategy = "Bundle deals, loyalty rewards"
        elif avg_m >= 500:
            persona = "💰 Big Spenders"
            strategy = "Increase frequency, VIP treatment"
        else:
            persona = "🌿 Potential Growth"
            strategy = "Engagement campaigns, education"
        
        personas[cluster_id] = {
            'name': persona,
            'size': len(segment),
            'avg_recency': avg_r,
            'avg_frequency': avg_f,
            'avg_monetary': avg_m,
            'avg_clv': avg_clv,
            'strategy': strategy
        }
    
    return personas

# ==========================================
# 4. MAIN APPLICATION LOGIC
# ==========================================
if uploaded_file and 'run_analysis' in locals() and run_analysis:
    with st.spinner("🔄 Processing data and building models..."):
        
        # --- 1. Load Data ---
        df = load_and_process(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} transactions from {df['CustomerID'].nunique():,} customers")
        
        # --- 2. Feature Engineering ---
        rfm_features = engineer_rfm_features(df)
        
        if use_nlp and 'Description' in df.columns:
            nlp_features = extract_nlp_features(df)
            if not nlp_features.empty:
                features = rfm_features.join(nlp_features, how='inner')
                st.success(f"✅ Added {nlp_features.shape[1]-1} Semantic Features (Transformer + PCA)")
            else:
                features = rfm_features
                st.warning("⚠️ NLP features skipped (insufficient text data)")
        else:
            features = rfm_features
        
        # --- 3. Clustering ---
        labels, X_scaled = run_clustering(features, algorithm, n_clusters)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1 and unique_labels[0] == -1:
             st.error("⚠️ Clustering found only 'Noise' (Outliers). Try switching to K-Means or adding more data.")
        else:
             st.success(f"✅ {algorithm} clustering: {len(np.unique(labels[labels != -1]))} segments identified")
        
        # --- 4. Modeling & CLV ---
        predictions, churn_model, feat_importance = train_churn_model(rfm_features, labels)
        predictions = estimate_clv(predictions)
        
        # --- 5. Group Personas ---
        personas = create_personas(predictions)
        st.success(f"✅ Generated {len(personas)} customer personas")
        
        st.markdown("---")
        
        # ==========================================
        # 5. DASHBOARD TABS
        # ==========================================
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Segments", "🔮 Predictions", "🧠 Model Insights"])
        
        # --- TAB 1: OVERVIEW ---
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Customers", f"{len(features):,}")
            with col2: st.metric("Segments Found", len(personas))
            with col3: st.metric("Avg CLV", f"${predictions['estimated_clv'].mean():,.0f}")
            with col4: st.metric("High Churn Risk", f"{(predictions['churn_probability'] > 0.6).sum():,}")
            
            st.markdown("### 🎯 Segment Performance Matrix")
            
            if personas:
                persona_rows = []
                for cid, pdata in personas.items():
                    persona_rows.append({
                        'Segment': f"Segment {cid}",
                        'Persona': pdata['name'],
                        'Customers': pdata['size'],
                        'Recency': pdata['avg_recency'],
                        'Monetary': pdata['avg_monetary']
                    })
                
                df_viz = pd.DataFrame(persona_rows)
                
                fig = go.Figure()
                for idx, row in df_viz.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['Recency']], y=[row['Monetary']],
                        mode='markers+text',
                        marker=dict(size=max(20, row['Customers']/2), opacity=0.7),
                        text=row['Persona'], textposition='top center',
                        name=row['Persona'],
                        hovertemplate=f"<b>{row['Persona']}</b><br>Size: {row['Customers']}<br>Rev: ${row['Monetary']:,.0f}"
                    ))
                
                fig.update_layout(
                    title='Segments: Recency vs Revenue (Bubble Size = Population)',
                    xaxis_title='Recency (Days)', yaxis_title='Monetary Value ($)',
                    template='plotly_dark', height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No segments found to visualize.")
        
        # --- TAB 2: PERSONAS (CLUSTER VIEW) ---
        with tab2:
            st.markdown("### 👑 Customer Personas (Group View)")
            for cid, data in personas.items():
                with st.expander(f"{data['name']} - Segment {cid}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Count", data['size'])
                    c2.metric("Recency", f"{data['avg_recency']:.0f} days")
                    c3.metric("Spend", f"${data['avg_monetary']:,.0f}")
                    c4.metric("CLV", f"${data['avg_clv']:,.0f}")
                    st.info(f"💡 Strategy: {data['strategy']}")
            
            st.markdown("### 🌐 3D Visualization")
            if len(features) > 5:
                embedding = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(features)-1)).fit_transform(X_scaled)
                viz_df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
                viz_df['Segment'] = [personas.get(l, {}).get('name', 'Outlier') for l in labels]
                
                fig = px.scatter_3d(viz_df, x='x', y='y', z='z', color='Segment', title='3D Cluster View (Semantic + Behavioral)')
                fig.update_traces(marker=dict(size=4, opacity=0.7))
                fig.update_layout(template='plotly_dark', height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # --- TAB 3: PREDICTIONS (INDIVIDUAL VIEW) ---
        with tab3:
            st.markdown("### 🔍 Customer Details (Individual View)")
            
            disp_df = predictions.reset_index().copy()
            
            # Rule-Based Logic for INDIVIDUAL Labels
            def get_individual_persona(row):
                r, f, m = row['recency_days'], row['frequency'], row['monetary_value']
                if r <= 40 and f >= 5 and m >= 1000: return "🏆 VIP Champion"
                if r > 150: return "💤 At-Risk/Dormant"
                if m >= 500: return "💰 Big Spender"
                if f >= 5: return "🔄 Frequent Buyer"
                if r <= 60: return "💎 Loyal Customer"
                return "🌿 Potential Growth"

            disp_df['Customer Status'] = disp_df.apply(get_individual_persona, axis=1)
            disp_df['Segment Group'] = [personas.get(l, {}).get('name', 'Outlier') for l in disp_df['cluster']]
            
            disp_df = disp_df[['CustomerID', 'Customer Status', 'Segment Group', 'churn_probability', 'estimated_clv', 'recency_days', 'monetary_value']]
            
            st.dataframe(disp_df.style.format({'churn_probability': '{:.1%}', 'estimated_clv': '${:,.2f}', 'monetary_value': '${:,.2f}'}), use_container_width=True)
            
            csv = disp_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")
            
        # --- TAB 4: INSIGHTS ---
        with tab4:
            st.markdown("### 🧠 Model Explainability")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Churn Predictors**")
                fig = px.bar(feat_importance, x='importance', y='feature', orientation='h', title='Feature Importance')
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Segment Size**")
                if personas:
                     names = [d['name'] for d in personas.values()]
                     sizes = [d['size'] for d in personas.values()]
                     fig = go.Figure(data=[go.Pie(labels=names, values=sizes, hole=0.3)])
                     fig.update_layout(template='plotly_dark')
                     st.plotly_chart(fig, use_container_width=True)

elif not uploaded_file:
    st.info("👈 Please upload your sales CSV file in the sidebar to start.")
