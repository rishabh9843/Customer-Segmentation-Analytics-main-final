
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

# --- STABLE IMAGE URLS ---
EDA_ICON_URL = "https://cdn-icons-png.flaticon.com/512/1998/1998557.png" 
SEGMENT_ICON_URL = "https://cdn-icons-png.flaticon.com/512/8956/8956264.png" 
SIMULATOR_ICON_URL = "https://cdn-icons-png.flaticon.com/512/5261/5261273.png"
# --- PREMIUM UI STYLING (CSS) ---
def load_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto:wght@400;500&display=swap');
            .main { background: #0D1117; color: #e0e0e0; font-family: 'Roboto', sans-serif; }
            [data-testid="stSidebar"] {
                background-color: rgba(15, 20, 35, 0.7); backdrop-filter: blur(15px);
                border-right: 1px solid rgba(0, 255, 255, 0.2);
            }
            .neon-title { font-family: 'Orbitron', sans-serif; color: #fff; text-align: center; text-shadow: 0 0 7px #00ffff, 0 0 10px #00ffff, 0 0 42px #00bfff; }
            .neon-title-sidebar { font-family: 'Orbitron', sans-serif; color: #fff; text-shadow: 0 0 5px #00bfff; }
            .stButton>button {
                color: #ffffff; background: linear-gradient(45deg, #0077b6 0%, #00bfff 100%);
                border: 1px solid #00ffff; border-radius: 8px; padding: 12px 28px; font-weight: 500;
                transition: all 0.3s ease-in-out; box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            }
            .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 30px rgba(0, 255, 255, 0.8); }
            .custom-card {
                background: rgba(20, 25, 40, 0.7); backdrop-filter: blur(10px); border-radius: 1rem;
                padding: 1.5rem; border: 1px solid rgba(0, 255, 255, 0.2); transition: all 0.3s ease-in-out;
                text-align: center; cursor: pointer; height: 100%; display: flex; flex-direction: column; justify-content: center;
            }
            .custom-card:hover { transform: translateY(-10px); border-color: #00ffff; box-shadow: 0 0 30px rgba(0, 255, 255, 0.5); }
            .custom-card img { width: 80px; margin-bottom: 1rem; }
            .custom-card h3 { color: #FFFFFF; font-weight: 600; font-size: 1.25rem; }
            .custom-card p { color: #b0b0b0; font-size: 0.9rem; }
            .st-emotion-cache-1r6slb0 { display: none; } /* Hides underlying button */
        </style>
    """, unsafe_allow_html=True)

def create_clickable_card(title, text, image_url, page_target):
    card_html = f"""<div class="custom-card" onclick="document.getElementById('btn-{page_target}').click()"><img src="{image_url}"><h3>{title}</h3><p>{text}</p></div>"""
    if st.button(f"Go to {page_target}", key=f"btn-{page_target}"):
        st.session_state.page = page_target
        st.rerun()
    st.markdown(card_html, unsafe_allow_html=True)

load_css()

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
def render_sidebar():
    with st.sidebar:
        st.markdown('<h1 class="neon-title-sidebar">SegmentIQ v6.1</h1>', unsafe_allow_html=True)
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload Your Sales Data (CSV)", type="csv")
        st.subheader("Clustering Configuration")
        algorithm = st.selectbox("Algorithm", ('HDBSCAN', 'K-Means', 'Gaussian Mixture'))
        if algorithm in ['K-Means', 'Gaussian Mixture']: param = st.slider("Number of Segments (K)", 2, 15, 5, 1)
        else: param = st.slider("Minimum Segment Size", 5, 100, 30, 5)
        run_button = st.button("Run Full Analysis")
        return uploaded_file, algorithm, param, run_button

def render_homepage():
    st.markdown('<h1 class="neon-title">AI Customer Analytics Suite</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b0b0b0;'>The ultimate tool for turning sales data into strategic intelligence.</p>", unsafe_allow_html=True)
    _, col_cards, _ = st.columns([0.5, 3, 0.5])
    with col_cards:
        cols = st.columns(3)
        with cols[0]: create_clickable_card("Exploratory Analysis", "Get a deep profile of your dataset.", EDA_ICON_URL, 'eda')
        with cols[1]: create_clickable_card("Deep Dive Analytics", "Segment customers and analyze personas.", SEGMENT_ICON_URL, 'analysis')
        with cols[2]: create_clickable_card("What-If Simulator", "Predict churn by changing customer behavior.", SIMULATOR_ICON_URL, 'simulator')

def render_eda_page(df):
    st.markdown('<h1 class="neon-title">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Home"): st.session_state.page = 'home'; st.rerun()
    st.dataframe(df.head()); cols = st.columns(3)
    cols[0].metric("Transactions", f"{len(df):,}"); cols[1].metric("Customers", f"{df['CustomerID'].nunique():,}"); cols[2].metric("Revenue", f"${df['Sales'].sum():,.2f}")
    fig = px.histogram(df, x="Sales", nbins=50, title="Distribution of Transaction Value")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
    st.plotly_chart(fig, use_container_width=True)

def render_analysis_dashboard(results):
    st.markdown('<h1 class="neon-title">Deep Dive Analytics Dashboard</h1>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Home"): st.session_state.page = 'home'; st.rerun()
    personas_df = pd.DataFrame.from_dict(results['personas'], orient='index')
    tabs = st.tabs(["🏆 Personas", "📊 Segment Analytics", "🧠 Model Insights", "🔮 Predictions & Lookup"])
    with tabs[0]:
        report = ""
        for cid, data in results['personas'].items(): report += f"## {data['persona']} (Segment {cid})\n- Customers: {data['size']:,}\n- Avg Recency: {data['avg_recency']:.0f} days\n\n"
        st.download_button("📥 Download Personas Report", report, "personas_report.txt")
        for cid, data in results['personas'].items():
            with st.expander(f"**{data['persona']}** (Segment {cid})"):
                cols = st.columns(4); cols[0].metric("Count", f"{data['size']:,}"); cols[1].metric("Avg. Recency", f"{data['avg_recency']:.0f}d"); cols[2].metric("Avg. Frequency", f"{data['avg_frequency']:.1f}x"); cols[3].metric("Avg. Spend", f"${data['avg_monetary']:,.2f}")
    with tabs[1]:
        sub_tabs = st.tabs(["Performance Matrix", "Behavior Patterns", "Segment Comparison"])
        with sub_tabs[0]: st.plotly_chart(px.scatter(personas_df, x="avg_recency", y="avg_monetary", size="size", color="persona", hover_name="persona", size_max=60, title="Segment Performance Matrix"), use_container_width=True)
        with sub_tabs[1]:
            behavior_df = results['predictions_df']; behavior_df['persona'] = [results['personas'].get(l, {}).get('persona', 'Outlier') for l in behavior_df['cluster']]
            st.plotly_chart(px.scatter(behavior_df, x='frequency', y='monetary_value', color='persona', title="Purchase Behavior Patterns"), use_container_width=True)
        with sub_tabs[2]:
            c1, c2 = st.columns(2); seg_list = personas_df['persona'].tolist()
            seg_a = c1.selectbox("Select Segment A", seg_list, index=0)
            seg_b = c2.selectbox("Select Segment B", seg_list, index=1 if len(seg_list) > 1 else 0)
            st.dataframe(pd.DataFrame([personas_df[personas_df['persona'] == seg_a].iloc[0], personas_df[personas_df['persona'] == seg_b].iloc[0]]))
    with tabs[2]:
        st.header("Model & Feature Insights")
        sub_tabs = st.tabs(["Feature Importance", "3D Cluster Visualizer", "Statistical Metrics"])
        with sub_tabs[0]: st.plotly_chart(px.bar(results['feature_importance'], x='importance', y='feature', orientation='h', title='Churn Prediction Feature Importance'), use_container_width=True)
        with sub_tabs[1]:
            embedding = umap.UMAP(n_components=3, random_state=42).fit_transform(results['features_scaled'])
            viz_df = pd.DataFrame(embedding, columns=['x', 'y', 'z']); viz_df['Persona'] = [results['personas'].get(l, {}).get('persona', 'Outlier') for l in results['labels']]
            st.plotly_chart(px.scatter_3d(viz_df, x='x', y='y', z='z', color='Persona', title="Interactive 3D Customer Segments", color_discrete_map={'Outlier': 'grey'}), use_container_width=True)
        with sub_tabs[2]:
            metrics = results.get('metrics', {}); st.metric("Silhouette Score", f"{metrics.get('silhouette', 0):.3f}"); st.metric("Calinski-Harabasz Score", f"{metrics.get('calinski_harabasz', 0):,.0f}"); st.metric("Davies-Bouldin Score", f"{metrics.get('davies_bouldin', 0):.3f}")
    with tabs[3]:
        st.header("Predictions & Customer Lookup")
        df_display = results['predictions_df'].reset_index(); df_display['Persona'] = [results['personas'].get(l, {}).get('persona', 'Outlier') for l in df_display['cluster']]
        search_id = st.text_input("Search by CustomerID"); 
        if search_id: df_display = df_display[df_display['CustomerID'].str.contains(search_id, case=False)]
        st.dataframe(df_display[['CustomerID', 'Persona', 'churn_probability', 'recency_days', 'frequency', 'monetary_value']])

# <<< THIS IS THE CORRECTED SIMULATOR FUNCTION
def render_simulator_page(results, features_df):
    st.markdown('<h1 class="neon-title">"What-If" Churn Simulator</h1>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Home"): st.session_state.page = 'home'; st.rerun()
    
    churn_model = results['churn_model']
    
    customer_list = features_df.index.tolist()
    selected_customer = st.selectbox("Select a Customer to Simulate", customer_list)
    
    customer_data = features_df.loc[selected_customer]
    
    st.subheader("Adjust Customer Metrics")
    new_recency = st.slider("Recency (Days)", 0, 365, int(customer_data['recency_days']))
    new_frequency = st.slider("Frequency (Purchases)", 1, 100, int(customer_data['frequency']))
    new_monetary = st.slider("Monetary Value ($)", 0, 10000, int(customer_data['monetary_value']))
    
    # Create a DataFrame for prediction, ensuring columns match the model's training data
    simulated_data = pd.DataFrame([{
        'recency_days': new_recency,
        'frequency': new_frequency,
        'monetary_value': new_monetary
    }])
    
    # Predict churn
    churn_prob = churn_model.predict_proba(simulated_data)[0][1]
    
    st.subheader("Predicted Churn Risk")
    st.progress(churn_prob)
    st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"{(churn_prob - customer_data['churn_probability']):.1%}", delta_color="inverse")
# >>> END OF CORRECTION

# --- Main App Logic ---
if __name__ == "__main__":
    uploaded_file, algorithm, param, run_button = render_sidebar()

    if run_button and uploaded_file:
        st.session_state.analysis_run = True; st.session_state.page = 'analysis'; st.rerun()

    if st.session_state.page == 'home': render_homepage()
    elif st.session_state.page == 'eda':
        if uploaded_file: render_eda_page(preprocess_data(load_data(uploaded_file)))
        else: st.warning("Please upload a file to perform EDA."); st.button("⬅️ Back", on_click=lambda: st.session_state.update(page='home'))
    elif st.session_state.page in ['analysis', 'simulator']:
        if not st.session_state.analysis_run:
            st.warning("Please run a full analysis from the sidebar first."); st.button("⬅️ Back", on_click=lambda: st.session_state.update(page='home'))
        else:
            if 'personas' not in st.session_state.results:
                with st.spinner("Running advanced AI analysis... Please wait."):
                    features_df = engineer_features(preprocess_data(load_data(uploaded_file)))
                    n_clusters = param if algorithm in ['K-Means', 'Gaussian Mixture'] else 5
                    min_size = param if algorithm == 'HDBSCAN' else 30
                    labels, scaled_features, metrics = run_clustering(features_df, algorithm, n_clusters, min_size)
                    predictions, churn_model, feat_imp = train_predictive_models(features_df, labels)
                    st.session_state.results = {'personas': generate_personas(predictions), 'labels': labels, 'features_scaled': scaled_features, 'predictions_df': predictions, 'metrics': metrics, 'churn_model': churn_model, 'feature_importance': feat_imp}
            
            if st.session_state.page == 'analysis': render_analysis_dashboard(st.session_state.results)

            elif st.session_state.page == 'simulator': render_simulator_page(st.session_state.results, st.session_state.results['predictions_df'])
