import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from collections import Counter
import hdbscan
import lightgbm as lgb
import umap

# Page config
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
st.markdown("**AI-powered customer insights using ensemble clustering, Keyword Analysis, and predictive modeling**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type="csv")
    algorithm = st.selectbox("Clustering Algorithm", ['K-Means', 'HDBSCAN'])
    n_clusters = st.slider("Number of Segments", 2, 10, 5) if algorithm == 'K-Means' else None
    use_nlp = st.checkbox("Enable NLP Features", value=True, help="Extract keyword features from descriptions")
    
    if uploaded_file:
        run_analysis = st.button("🚀 Run Analysis", use_container_width=True)
    else:
        st.info("Upload CSV to begin")

# Main functions
@st.cache_data
def load_and_process(file):
    """Load and preprocess data"""
    df = pd.read_csv(file)
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    return df

@st.cache_data
def engineer_rfm_features(df):
    """Create RFM features"""
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
    Simplified NLP: Counts how often customers buy items with specific top keywords.
    Easier to explain than TF-IDF/SVD.
    """
    if 'Description' not in df.columns:
        return pd.DataFrame()
    
    # 1. Clean Text
    df_clean = df.copy()
    df_clean['Description'] = df_clean['Description'].fillna('').astype(str).str.lower()
    
    # 2. Find the Top 5 most common words across the ENTIRE store
    # We combine all text, split into words, and count them
    all_text = ' '.join(df_clean['Description'].tolist())
    words = all_text.split()
    
    # Filter out boring words (stop words)
    stop_words = ['of', 'the', 'in', 'and', 'set', 'a', 'white', 'red', 'blue', 'pink', 'black', 'pack']
    interesting_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Get top 5 most frequent words (e.g., 'bag', 'box', 'glass', 'sign', 'holder')
    if not interesting_words:
        return pd.DataFrame() # Return empty if no words found
        
    top_5_words = [word for word, count in Counter(interesting_words).most_common(5)]
    
    # 3. Score each customer based on these 5 words
    def count_keywords(customer_descriptions):
        full_text = ' '.join(customer_descriptions)
        # Create a dictionary of counts for the top 5 words
        return pd.Series({f'keyword_{w}': full_text.count(w) for w in top_5_words})

    # Apply this counting logic to every customer
    nlp_features = df_clean.groupby('CustomerID')['Description'].apply(count_keywords).unstack()
    
    # 4. Add Product Variety (Simple diversity metric)
    nlp_features['unique_products_count'] = df_clean.groupby('CustomerID')['StockCode'].nunique()
    
    return nlp_features

def run_clustering(features_df, algo, k):
    """Perform ensemble clustering"""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    if algo == 'K-Means':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        model = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean')
    
    labels = model.fit_predict(X_scaled)
    return labels, X_scaled

def train_churn_model(features_df, labels):
    """Train LightGBM churn prediction model"""
    df = features_df.copy()
    df['cluster'] = labels
    
    # Define churn (high recency = at risk)
    churn_threshold = df['recency_days'].quantile(0.75)
    df['is_churn'] = (df['recency_days'] > churn_threshold).astype(int)
    
    # Use only core RFM features for prediction
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
    """Estimate Customer Lifetime Value"""
    predictions_df['estimated_clv'] = (
        predictions_df['monetary_value'] * (1 - predictions_df['churn_probability']) * (predictions_df['frequency'] / 12)
    )
    return predictions_df

def create_personas(predictions_df):
    """Generate customer personas"""
    personas = {}
    for cluster_id in sorted(predictions_df['cluster'].unique()):
        if cluster_id == -1:
            continue
        
        segment = predictions_df[predictions_df['cluster'] == cluster_id]
        avg_r = segment['recency_days'].mean()
        avg_f = segment['frequency'].mean()
        avg_m = segment['monetary_value'].mean()
        avg_clv = segment['estimated_clv'].mean()
        
        # Assign persona based on RFM
        if avg_r <= 30 and avg_f >= 10 and avg_m >= 1000:
            persona = "🏆 VIP Champions"
            strategy = "Exclusive benefits, personalized service"
        elif avg_r <= 60 and avg_f >= 5:
            persona = "💎 Loyal Customers"
            strategy = "Upsell premium, referral programs"
        elif avg_r > 180:
            persona = "💤 At-Risk/Dormant"
            strategy = "Win-back campaigns, special offers"
        elif avg_f >= 8:
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

# Main execution
if uploaded_file and 'run_analysis' in locals() and run_analysis:
    with st.spinner("🔄 Processing data and building models..."):
        
        # Pipeline
        df = load_and_process(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} transactions from {df['CustomerID'].nunique():,} customers")
        
        # RFM Features
        rfm_features = engineer_rfm_features(df)
        st.success(f"✅ Engineered RFM features for {len(rfm_features):,} customers")
        
        # NLP Features (Simplified)
        if use_nlp and 'Description' in df.columns:
            nlp_features = extract_nlp_features(df)
            if not nlp_features.empty:
                # Combine RFM + NLP
                features = rfm_features.join(nlp_features, how='inner')
                st.success(f"✅ Added {nlp_features.shape[1]} Keyword features (Top Words + Variety)")
            else:
                features = rfm_features
                st.warning("⚠️ NLP features skipped (insufficient text data)")
        else:
            features = rfm_features
            if use_nlp:
                st.info("ℹ️ NLP disabled (no Description column found)")
        
        st.success(f"✅ Total features: {features.shape[1]}")
        
        # Clustering
        labels, X_scaled = run_clustering(features, algorithm, n_clusters)
        st.success(f"✅ {algorithm} clustering: {len(np.unique(labels[labels != -1]))} segments identified")
        
        # Predictive modeling
        predictions, churn_model, feat_importance = train_churn_model(rfm_features, labels)
        st.success("✅ LightGBM churn model trained")
        
        # CLV estimation
        predictions = estimate_clv(predictions)
        st.success("✅ Customer Lifetime Value estimated")
        
        # Customer personas
        personas = create_personas(predictions)
        st.success(f"✅ Generated {len(personas)} customer personas")
        
        st.markdown("---")
        
        # Results
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Segments", "🔮 Predictions", "🧠 Model Insights"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", f"{len(features):,}")
            with col2:
                st.metric("Segments Found", len(personas))
            with col3:
                avg_clv = predictions['estimated_clv'].mean()
                st.metric("Avg CLV", f"${avg_clv:,.0f}")
            with col4:
                high_risk = (predictions['churn_probability'] > 0.6).sum()
                st.metric("High Churn Risk", f"{high_risk:,}")
            
            st.markdown("### 🎯 Segment Performance Matrix")
            
            try:
                # Create dataframe from personas dictionary
                persona_rows = []
                for cid, pdata in personas.items():
                    persona_rows.append({
                        'Segment': f"Segment {cid}",
                        'Persona': pdata['name'],
                        'Customers': pdata['size'],
                        'Recency': pdata['avg_recency'],
                        'Monetary': pdata['avg_monetary']
                    })
                
                if persona_rows:
                    df_viz = pd.DataFrame(persona_rows)
                    
                    fig = go.Figure()
                    
                    for idx, row in df_viz.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['Recency']],
                            y=[row['Monetary']],
                            mode='markers+text',
                            marker=dict(size=max(20, row['Customers']/5), opacity=0.7),
                            text=row['Persona'],
                            textposition='top center',
                            name=row['Persona'],
                            hovertemplate=f"<b>{row['Persona']}</b><br>" +
                                          f"Customers: {row['Customers']}<br>" +
                                          f"Recency: {row['Recency']:.0f} days<br>" +
                                          f"Monetary: ${row['Monetary']:,.0f}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title='Customer Segments: Recency vs Revenue',
                        xaxis_title='Recency (days)',
                        yaxis_title='Monetary Value ($)',
                        template='plotly_dark',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No segments to visualize")
                    
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
        
        with tab2:
            st.markdown("### 👑 Customer Personas")
            for cid, data in personas.items():
                with st.expander(f"{data['name']} - Segment {cid}", expanded=True):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Customers", f"{data['size']:,}")
                    col2.metric("Avg Recency", f"{data['avg_recency']:.0f}d")
                    col3.metric("Avg Frequency", f"{data['avg_frequency']:.1f}")
                    col4.metric("Avg Spend", f"${data['avg_monetary']:,.0f}")
                    col5.metric("Avg CLV", f"${data['avg_clv']:,.0f}")
                    
                    st.markdown(f"**💡 Strategy:** {data['strategy']}")
            
            st.markdown("### 🌐 3D Cluster Visualization (UMAP)")
            embedding = umap.UMAP(n_components=3, random_state=42, n_neighbors=15).fit_transform(X_scaled)
            viz_df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
            viz_df['Segment'] = [personas.get(l, {}).get('name', 'Outlier') for l in labels]
            
            fig = px.scatter_3d(
                viz_df, 
                x='x', y='y', z='z', 
                color='Segment',
                title='3D Customer Segmentation (UMAP Dimensionality Reduction)'
            )
            fig.update_traces(marker=dict(size=4, opacity=0.7))
            fig.update_layout(template='plotly_dark', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 Customer Churn Predictions")
            
            display_df = predictions.reset_index()
            display_df['Persona'] = [personas.get(l, {}).get('name', 'Unknown') for l in display_df['cluster']]
            display_df = display_df[[
                'CustomerID', 'Persona', 'churn_probability', 'estimated_clv',
                'recency_days', 'frequency', 'monetary_value'
            ]]
            
            # Search
            search = st.text_input("🔍 Search Customer ID")
            if search:
                display_df = display_df[display_df['CustomerID'].str.contains(search, case=False)]
            
            # Display
            st.dataframe(
                display_df.style.format({
                    'churn_probability': '{:.1%}',
                    'estimated_clv': '${:,.2f}',
                    'monetary_value': '${:,.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "customer_predictions.csv", "text/csv")
        
        with tab4:
            st.markdown("### 🧠 Model Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Feature Importance (Churn Model)")
                fig = px.bar(
                    feat_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='LightGBM Feature Importance',
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Segment Distribution")
                
                try:
                    # Create simple lists for pie chart
                    segment_names = [data['name'] for data in personas.values()]
                    segment_sizes = [data['size'] for data in personas.values()]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=segment_names,
                        values=segment_sizes,
                        hole=0.3
                    )])
                    
                    fig.update_layout(
                        title='Customer Distribution by Segment',
                        template='plotly_dark',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Pie chart error: {str(e)}")
            
            # Summary
            st.markdown("### 📈 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔧 Features Used:**")
                st.markdown(f"- RFM Features: 3")
                if use_nlp and not nlp_features.empty:
                    st.markdown(f"- Keyword Features: {nlp_features.shape[1]}")
                st.markdown(f"- Total: {features.shape[1]}")
            
            with col2:
                st.markdown("**🤖 ML Techniques:**")
                st.markdown(f"- Clustering: {algorithm}")
                st.markdown("- Churn Model: LightGBM")
                st.markdown("- Dim Reduction: UMAP")
                if use_nlp:
                    st.markdown("- NLP: Keyword Frequency")
            
            with col3:
                st.markdown("**💼 Business Value:**")
                total_clv = predictions['estimated_clv'].sum()
                st.markdown(f"- Total Portfolio CLV: ${total_clv:,.0f}")
                st.markdown(f"- High-Risk Customers: {high_risk:,}")
                st.markdown(f"- Segments Identified: {len(personas)}")

elif not uploaded_file:
    st.info("👈 Upload your sales data CSV to get started")
    
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    **Full-stack data pipeline with ensemble ML & Keyword Analysis for customer analytics**
    
    **Key Technologies:**
    - **Data Processing:** Pandas, NumPy for feature engineering (RFM analysis)
    - **ML Models:** Ensemble Clustering (K-Means, HDBSCAN) for segmentation
    - **Predictive Analytics:** LightGBM for churn prediction & CLV estimation
    - **NLP:** Keyword Frequency Analysis for product categorization
    - **Visualization:** Plotly 3D, UMAP dimensionality reduction
    
    **Business Impact:**
    - Automated customer segmentation with 5-10 distinct personas
    - Predictive churn modeling for retention strategies
    - CLV estimation for strategic planning
    - Real-time analytics dashboard for decision-making
    - Product preference analysis via Keyword Trends
    """)
