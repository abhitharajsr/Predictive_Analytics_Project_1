"""
🔮 Telecom Customer Churn Prediction — Streamlit Dashboard
===========================================================
A premium, interactive web application for predicting customer churn
in telecommunications companies using Machine Learning.

Author: Capstone Project — Predictive Analytics
Dataset: IBM Telco Customer Churn (Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Import Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }
    
    /* ─── Metric Cards ─── */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 16px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.35);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h1 {
        margin: 8px 0 0 0;
        font-size: 36px;
        font-weight: 800;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        box-shadow: 0 8px 32px rgba(0, 176, 155, 0.25);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.25);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.25);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.25);
    }
    
    /* ─── Prediction Result ─── */
    .prediction-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        border-radius: 16px;
        padding: 32px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.3);
    }
    .prediction-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        border-radius: 16px;
        padding: 32px;
        color: #333;
        text-align: center;
        box-shadow: 0 8px 32px rgba(247, 151, 30, 0.3);
    }
    .prediction-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 32px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
    }
    
    /* ─── Section Headers ─── */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    
    /* ─── Info Box ─── */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #667eea;
        margin: 16px 0;
    }
    
    /* ─── Hide Streamlit Elements ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ─── Divider ─── */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #ff6584, #00d2ff);
        border: none;
        border-radius: 2px;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

COLOR_PALETTE = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
CHURN_COLORS = {'No': '#43e97b', 'Yes': '#f5576c'}


# ─── Data Loading ────────────────────────────────────────
@st.cache_data
def load_raw_data():
    """Load and minimally clean the raw dataset for EDA."""
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
    df['TotalCharges'].fillna(0, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df


@st.cache_resource
def load_trained_model():
    """Load the trained model and metadata."""
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'model_metadata.pkl'))
        all_results = joblib.load(os.path.join(MODELS_DIR, 'all_results.pkl'))
        return model, scaler, metadata, all_results
    except FileNotFoundError:
        return None, None, None, None


# ─── Helper Functions ────────────────────────────────────
def metric_card(title, value, card_class="metric-card"):
    return f"""
    <div class="{card_class}">
        <h3>{title}</h3>
        <h1>{value}</h1>
    </div>
    """


def make_gauge(value, title="Churn Probability"):
    """Create a beautiful gauge chart."""
    color = '#43e97b' if value < 0.3 else ('#f7971e' if value < 0.6 else '#ff416c')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#333', 'family': 'Inter'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#333', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#666'},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#f0f2f6',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(67, 233, 123, 0.15)'},
                {'range': [30, 60], 'color': 'rgba(247, 151, 30, 0.15)'},
                {'range': [60, 100], 'color': 'rgba(255, 65, 108, 0.15)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📡 Churn Predictor")
    st.markdown("---")
    
    page = st.radio(
        "**Navigation**",
        ["🏠 Overview", "📊 Exploratory Analysis", "🔮 Predict Churn", "📈 Model Performance", "🧠 SHAP Explainability", "💡 Retention Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; opacity: 0.7; font-size: 12px;'>
        <p>Built with ❤️ using Streamlit</p>
        <p>Dataset: IBM Telco Churn</p>
        <p>© 2026 Capstone Project</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Load SHAP Data ──────────────────────────────────────
@st.cache_resource
def load_shap_data():
    """Load precomputed SHAP values."""
    try:
        shap_data = joblib.load(os.path.join(MODELS_DIR, 'shap_values.pkl'))
        return shap_data
    except FileNotFoundError:
        return None


# ─── Load Data ───────────────────────────────────────────
df = load_raw_data()
model, scaler, metadata, all_results = load_trained_model()
shap_data = load_shap_data()


# ═══════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="section-header">📡 Telecom Customer Churn Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Predicting which customers are likely to leave — *powered by Machine Learning*")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # KPI Cards
    total = len(df)
    churned = df['Churn'].value_counts().get('Yes', 0)
    retained = total - churned
    churn_rate = churned / total * 100
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Total Customers", f"{total:,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Churned", f"{churned:,}", "metric-card metric-card-red"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Retained", f"{retained:,}", "metric-card metric-card-green"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Churn Rate", f"{churn_rate:.1f}%", "metric-card metric-card-orange"), unsafe_allow_html=True)
    
    st.markdown("")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn Distribution
        churn_counts = df['Churn'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            hole=0.55,
            marker_colors=[CHURN_COLORS['No'], CHURN_COLORS['Yes']],
            textfont_size=14,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
        )])
        fig.update_layout(
            title=dict(text='Churn Distribution', font=dict(size=18, family='Inter')),
            height=400,
            showlegend=True,
            legend=dict(x=0.35, y=-0.05, orientation='h'),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            annotations=[dict(text=f'<b>{churn_rate:.1f}%</b><br>Churn', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn by Contract Type
        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig = px.bar(
            contract_churn, x='Contract', y='Count', color='Churn',
            barmode='group', color_discrete_map=CHURN_COLORS,
            title='Churn by Contract Type'
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font_size=18,
            xaxis_title='', yaxis_title='Number of Customers',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure Distribution by Churn
        fig = go.Figure()
        for churn_val, color in CHURN_COLORS.items():
            subset = df[df['Churn'] == churn_val]
            fig.add_trace(go.Histogram(
                x=subset['tenure'], name=f'Churn: {churn_val}',
                marker_color=color, opacity=0.7,
                nbinsx=30
            ))
        fig.update_layout(
            title=dict(text='Tenure Distribution by Churn', font=dict(size=18)),
            barmode='overlay', height=380,
            xaxis_title='Tenure (months)', yaxis_title='Count',
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly Charges Distribution by Churn
        fig = go.Figure()
        for churn_val, color in CHURN_COLORS.items():
            subset = df[df['Churn'] == churn_val]
            fig.add_trace(go.Box(
                y=subset['MonthlyCharges'], name=f'Churn: {churn_val}',
                marker_color=color, boxmean='sd'
            ))
        fig.update_layout(
            title=dict(text='Monthly Charges by Churn', font=dict(size=18)),
            height=380,
            yaxis_title='Monthly Charges ($)',
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💡 Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box">
            <b>📋 Month-to-Month Risk</b><br>
            Customers on month-to-month contracts are <b>3x more likely</b> to churn than those on yearly contracts.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
            <b>⏱️ Tenure Matters</b><br>
            New customers (0-12 months) have the <b>highest churn rate</b>. Loyalty programs can help retain them.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-box">
            <b>🌐 Fiber Optic</b><br>
            Fiber optic users show <b>higher churn rates</b> — possibly due to premium pricing without matching service quality.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 2: EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("Deep dive into customer data patterns and churn drivers")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview", "📊 Categorical Analysis",
        "📈 Numerical Analysis", "🔥 Correlation"
    ])
    
    # --- Tab 1: Data Overview ---
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 Dataset Shape")
            st.info(f"**Rows:** {df.shape[0]:,}  |  **Columns:** {df.shape[1]}")
            
            st.markdown("#### 📊 Column Types")
            dtype_counts = df.dtypes.value_counts()
            st.write(dtype_counts)
            
            st.markdown("#### ❌ Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) == 0:
                st.success("No missing values! ✅")
            else:
                st.write(missing)
        
        with col2:
            st.markdown("#### 📈 Descriptive Statistics")
            st.dataframe(df.describe().round(2), use_container_width=True)
        
        st.markdown("#### 🔍 Sample Data (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)
    
    # --- Tab 2: Categorical Analysis ---
    with tab2:
        st.markdown("#### Churn Rate by Category")
        
        cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']
        
        selected_cat = st.selectbox("Select Feature", cat_cols, index=cat_cols.index('Contract'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count plot
            cat_counts = df.groupby([selected_cat, 'Churn']).size().reset_index(name='Count')
            fig = px.bar(
                cat_counts, x=selected_cat, y='Count', color='Churn',
                barmode='group', color_discrete_map=CHURN_COLORS,
                title=f'Distribution: {selected_cat}'
            )
            fig.update_layout(
                height=420, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'), title_font_size=16,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn rate per category
            churn_rate_cat = df.groupby(selected_cat)['Churn'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            churn_rate_cat.columns = [selected_cat, 'Churn Rate (%)']
            
            fig = px.bar(
                churn_rate_cat, x=selected_cat, y='Churn Rate (%)',
                color='Churn Rate (%)',
                color_continuous_scale=['#43e97b', '#f7971e', '#ff416c'],
                title=f'Churn Rate by {selected_cat}'
            )
            fig.update_layout(
                height=420, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'), title_font_size=16,
                showlegend=False,
            )
            fig.add_hline(y=26.5, line_dash="dash", line_color="#667eea",
                         annotation_text="Overall Avg", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)
        
        # All categories churn rate grid
        st.markdown("#### 📊 Churn Rate Across All Categories")
        churn_rates = []
        for col in cat_cols:
            rates = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            for cat_val, rate in rates.items():
                churn_rates.append({'Feature': col, 'Category': str(cat_val), 'Churn Rate (%)': round(rate, 1)})
        
        churn_df = pd.DataFrame(churn_rates)
        churn_pivot = churn_df.pivot(index='Feature', columns='Category', values='Churn Rate (%)')
        
        fig = px.imshow(
            churn_pivot, text_auto='.1f',
            color_continuous_scale=['#43e97b', '#ffd200', '#ff416c'],
            title='Churn Rate Heatmap (%)',
            aspect='auto'
        )
        fig.update_layout(
            height=600, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=16,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Tab 3: Numerical Analysis ---
    with tab3:
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        selected_num = st.selectbox("Select Numerical Feature", num_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with churn overlay
            fig = px.histogram(
                df, x=selected_num, color='Churn',
                color_discrete_map=CHURN_COLORS,
                barmode='overlay', opacity=0.7,
                nbins=40, title=f'{selected_num} Distribution by Churn'
            )
            fig.update_layout(
                height=400, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'), title_font_size=16,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot
            fig = px.violin(
                df, x='Churn', y=selected_num, color='Churn',
                color_discrete_map=CHURN_COLORS,
                box=True, points='outliers',
                title=f'{selected_num} — Violin Plot'
            )
            fig.update_layout(
                height=400, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'), title_font_size=16,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("#### 📈 Feature Relationships")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-Axis", num_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-Axis", num_cols, index=1)
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color='Churn',
            color_discrete_map=CHURN_COLORS,
            opacity=0.5, title=f'{x_axis} vs {y_axis}',
            trendline='ols'
        )
        fig.update_layout(
            height=450, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=16,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Tab 4: Correlation ---
    with tab4:
        st.markdown("#### 🔥 Correlation Heatmap")
        
        # Prepare numeric data
        df_numeric = df.copy()
        df_numeric['Churn'] = df_numeric['Churn'].map({'Yes': 1, 'No': 0})
        df_numeric['SeniorCitizen'] = df_numeric['SeniorCitizen'].map({'Yes': 1, 'No': 0})
        
        # Select numeric columns
        numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()
        if 'customerID' in numeric_cols:
            numeric_cols.remove('customerID')
        
        corr = df_numeric[numeric_cols].corr()
        
        fig = px.imshow(
            corr, text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix',
            zmin=-1, zmax=1,
        )
        fig.update_layout(
            height=500, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=16,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target
        st.markdown("#### 🎯 Correlation with Churn (Target)")
        churn_corr = corr['Churn'].drop('Churn').sort_values(ascending=True)
        
        fig = go.Figure(go.Bar(
            x=churn_corr.values,
            y=churn_corr.index,
            orientation='h',
            marker_color=['#ff416c' if v > 0 else '#43e97b' for v in churn_corr.values],
        ))
        fig.update_layout(
            title='Correlation with Churn',
            height=350, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=16,
            xaxis_title='Correlation Coefficient',
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 3: PREDICT CHURN
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.markdown('<p class="section-header">🔮 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown("Enter customer details to predict their likelihood of churning")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ No trained model found! Please run the training pipeline first.")
        st.code("python src/model_training.py", language="bash")
        st.stop()
    
    st.markdown(f"**Active Model:** `{metadata['best_model_name']}`")
    
    # Input Form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", key="senior")
        partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="deps")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
    
    with col2:
        st.markdown("#### 📞 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="multi")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="protect")
    
    with col3:
        st.markdown("#### 💰 Account")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="stv")
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="smov")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paper")
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="payment")
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key="monthly")
        total = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                float(monthly * tenure), step=10.0, key="total")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Predict Button
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        # Build input dict
        input_data = {
            'gender': gender, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone,
            'MultipleLines': multi_lines, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly,
            'TotalCharges': total,
        }
        
        # Preprocess
        sys.path.insert(0, BASE_DIR)
        from src.data_preprocessing import preprocess_single_input
        
        try:
            X_input = preprocess_single_input(input_data, scaler, metadata['feature_names'])
            
            # Predict
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            churn_prob = probability[1]
            
            # Results
            st.markdown("")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Gauge Chart
                fig = make_gauge(churn_prob)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk Classification
                if churn_prob >= 0.6:
                    risk = "🔴 HIGH RISK"
                    css_class = "prediction-high"
                    advice = "Immediate intervention needed! Consider offering discounts, contract upgrades, or personalized retention offers."
                elif churn_prob >= 0.3:
                    risk = "🟡 MEDIUM RISK"
                    css_class = "prediction-medium"
                    advice = "Monitor closely. Proactive engagement and satisfaction surveys recommended."
                else:
                    risk = "🟢 LOW RISK"
                    css_class = "prediction-low"
                    advice = "Customer appears satisfied. Continue providing excellent service."
                
                st.markdown(f"""
                <div class="{css_class}">
                    <h2 style="margin:0; font-size: 32px;">{risk}</h2>
                    <p style="font-size: 18px; margin-top: 12px;">Churn Probability: <b>{churn_prob:.1%}</b></p>
                    <p style="font-size: 14px; margin-top: 8px; opacity: 0.9;">Prediction: <b>{'Will Churn' if prediction == 1 else 'Will Stay'}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box" style="margin-top: 16px;">
                    <b>💡 Recommended Action:</b><br>
                    {advice}
                </div>
                """, unsafe_allow_html=True)
            
            # Feature contribution breakdown
            st.markdown("### 📊 Key Factors for This Prediction")
            
            factors = []
            if contract == "Month-to-month":
                factors.append(("⚠️ Month-to-month contract", "high risk factor", "#ff416c"))
            elif contract == "Two year":
                factors.append(("✅ Two-year contract", "strong retention signal", "#43e97b"))
            
            if tenure <= 12:
                factors.append(("⚠️ Low tenure (< 12 months)", "new customers churn more", "#ff416c"))
            elif tenure >= 48:
                factors.append(("✅ Long tenure (48+ months)", "loyal customer", "#43e97b"))
            
            if internet == "Fiber optic":
                factors.append(("⚠️ Fiber optic internet", "higher churn rate segment", "#f7971e"))
            
            if payment == "Electronic check":
                factors.append(("⚠️ Electronic check payment", "correlated with churn", "#f7971e"))
            
            if monthly > 70:
                factors.append(("⚠️ High monthly charges", "price sensitivity risk", "#f7971e"))
            
            if tech_support == "No" and internet != "No":
                factors.append(("⚠️ No tech support", "support matters for retention", "#f7971e"))
            
            if security == "No" and internet != "No":
                factors.append(("⚠️ No online security", "add-on services reduce churn", "#f7971e"))
            
            if factors:
                for factor, desc, color in factors:
                    st.markdown(f"- **{factor}** — _{desc}_")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════
#  PAGE 4: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<p class="section-header">📈 Model Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Comparing all trained models side by side")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if metadata is None:
        st.error("⚠️ No trained model found! Please run the training pipeline first.")
        st.stop()
    
    # Best Model Highlight
    st.markdown(f"""
    <div class="metric-card metric-card-green" style="max-width: 500px;">
        <h3>🏆 Best Model</h3>
        <h1>{metadata['best_model_name']}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Table
    metrics_df = pd.DataFrame(metadata['metrics'])
    
    st.markdown("### 📊 Performance Comparison")
    
    # Styled dataframe
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    display_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]].copy()
    
    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Metrics Bar Chart
    metrics_for_plot = pd.DataFrame(metadata['metrics'])
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_cols = [c for c in metric_cols if c in metrics_for_plot.columns]
    
    fig = go.Figure()
    colors = ['#667eea', '#f093fb', '#43e97b', '#4facfe', '#fa709a']
    
    for i, metric in enumerate(available_cols):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_for_plot['Model'],
            y=metrics_for_plot[metric],
            marker_color=colors[i % len(colors)],
            opacity=0.85,
        ))
    
    fig.update_layout(
        title='Model Metrics Comparison',
        barmode='group', height=450,
        paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
        title_font_size=18, yaxis_range=[0, 1.05],
        xaxis_title='', yaxis_title='Score',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curves
    if all_results is not None:
        st.markdown("### 📈 ROC Curves")
        
        # Load test data for ROC
        y_test = None
        try:
            sys.path.insert(0, BASE_DIR)
            from src.data_preprocessing import full_preprocessing_pipeline
            _, X_test, _, y_test, _, _, _ = full_preprocessing_pipeline(DATA_PATH)
        except:
            pass
        
        if y_test is not None:
            from sklearn.metrics import roc_curve, auc
            
            fig = go.Figure()
            colors_roc = ['#667eea', '#f093fb', '#43e97b', '#4facfe', '#fa709a']
            
            for i, (name, data) in enumerate(all_results.items()):
                if data['y_prob'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
                    auc_score = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, name=f'{name} (AUC={auc_score:.3f})',
                        line=dict(color=colors_roc[i % len(colors_roc)], width=2.5),
                    ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name='Random',
                line=dict(color='gray', dash='dash', width=1),
            ))
            
            fig.update_layout(
                title='ROC Curves — All Models',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'), title_font_size=18,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    if all_results is not None and y_test is not None:
        st.markdown("### 🎯 Confusion Matrices")
        
        from sklearn.metrics import confusion_matrix
        
        model_names = list(all_results.keys())
        n_models = len(model_names)
        cols = st.columns(min(n_models, 3))
        
        for i, name in enumerate(model_names):
            with cols[i % 3]:
                cm = confusion_matrix(y_test, all_results[name]['y_pred'])
                
                fig = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x='Predicted', y='Actual'),
                    x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                    color_continuous_scale='RdPu',
                    title=name
                )
                fig.update_layout(
                    height=320, paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', size=11), title_font_size=14,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (for tree-based models)
    if all_results is not None and metadata is not None:
        st.markdown("### 🌟 Feature Importance")
        
        best_name = metadata['best_model_name']
        best_model_obj = all_results.get(best_name, {}).get('model')
        
        if best_model_obj is not None and hasattr(best_model_obj, 'feature_importances_'):
            importances = best_model_obj.feature_importances_
            features = metadata['feature_names']
            
            # Sort by importance
            indices = np.argsort(importances)[-20:]  # Top 20
            
            fig = go.Figure(go.Bar(
                x=importances[indices],
                y=[features[i] for i in indices],
                orientation='h',
                marker=dict(
                    color=importances[indices],
                    colorscale=['#667eea', '#f093fb', '#f5576c'],
                ),
            ))
            fig.update_layout(
                title=f'Top 20 Feature Importances — {best_name}',
                height=550, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'), title_font_size=18,
                xaxis_title='Importance',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {best_name}. Available for tree-based models.")


# ═══════════════════════════════════════════════════════════
#  PAGE 5: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.markdown('<p class="section-header">🧠 SHAP — Model Explainability</p>', unsafe_allow_html=True)
    st.markdown("Understanding *why* the model predicts churn — powered by SHapley Additive exPlanations")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if shap_data is None:
        st.error("⚠️ SHAP values not computed yet! Run: `python -m src.shap_analysis`")
        st.stop()
    
    shap_vals = shap_data['shap_values']
    feat_names = shap_data['feature_names']
    shap_model_name = shap_data['model_name']
    X_test_shap = shap_data['X_test']
    
    st.markdown(f"**Explainability Model:** `{shap_model_name}`")
    
    # Handle SHAP values format
    if hasattr(shap_vals, 'values'):
        shap_matrix = shap_vals.values
    else:
        shap_matrix = shap_vals
    
    # If 3D (binary classification), take positive class
    if len(shap_matrix.shape) == 3:
        shap_matrix = shap_matrix[:, :, 1]
    
    # ─── What is SHAP? ───
    with st.expander("ℹ️ What is SHAP?", expanded=False):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is a game-theory approach to explain ML predictions.
        
        - Each feature gets a **SHAP value** = its contribution to pushing the prediction from the average.
        - **Positive SHAP** → pushes toward **churn** (red) 
        - **Negative SHAP** → pushes toward **no churn** (blue)
        - The magnitude tells you **how much** the feature matters.
        
        Think of it like this: if the average churn probability is 26%, SHAP tells you *exactly* which features 
        pushed a particular customer's prediction up to 75% or down to 5%.
        """)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Global Importance", "🔍 Individual Explanations", "🔗 Feature Interactions"
    ])
    
    # ─── Tab 1: Global SHAP ───
    with tab1:
        st.markdown("#### 🌍 Global Feature Importance (Mean |SHAP|)")
        st.markdown("Which features matter most *across all customers*?")
        
        # Compute mean absolute SHAP
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1][:20]  # Top 20
        
        fig = go.Figure(go.Bar(
            x=mean_abs[sorted_idx][::-1],
            y=[feat_names[i] for i in sorted_idx][::-1],
            orientation='h',
            marker=dict(
                color=mean_abs[sorted_idx][::-1],
                colorscale=[[0, '#667eea'], [0.5, '#f093fb'], [1, '#f5576c']],
            ),
        ))
        fig.update_layout(
            title=f'Top 20 Features by SHAP Importance — {shap_model_name}',
            height=600, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=18,
            xaxis_title='Mean |SHAP Value|',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # SHAP Summary: Beeswarm-style as scatter
        st.markdown("#### 🐝 SHAP Summary (Beeswarm Plot)")
        st.markdown("Each dot = one customer. Color = feature value (red=high, blue=low). X-position = SHAP impact.")
        
        # Pick top 15 features for beeswarm
        top_n_bee = 15
        top_idx_bee = np.argsort(mean_abs)[::-1][:top_n_bee]
        
        scatter_data = []
        for rank, feat_idx in enumerate(top_idx_bee):
            shap_col = shap_matrix[:, feat_idx]
            feat_col = X_test_shap[:, feat_idx]
            n_pts = min(len(shap_col), 500)  # Limit points for performance
            sample_idx = np.random.choice(len(shap_col), n_pts, replace=False)
            for s_i in sample_idx:
                scatter_data.append({
                    'Feature': feat_names[feat_idx],
                    'SHAP Value': shap_col[s_i],
                    'Feature Value': feat_col[s_i],
                    'Rank': rank,
                })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        fig = px.scatter(
            scatter_df, x='SHAP Value', y='Feature', color='Feature Value',
            color_continuous_scale='RdBu_r',
            title='SHAP Summary — Feature Impact on Churn',
            category_orders={'Feature': [feat_names[i] for i in top_idx_bee]},
            opacity=0.6,
        )
        fig.update_layout(
            height=650, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=18,
            yaxis_title='', xaxis_title='SHAP Value (impact on churn prediction)',
        )
        fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    # ─── Tab 2: Individual Explanations ───
    with tab2:
        st.markdown("#### 🔍 Explain a Single Customer's Prediction")
        st.markdown("Select a test customer to see *why* the model made its prediction.")
        
        n_samples = min(len(X_test_shap), 500)
        customer_idx = st.slider("Select Customer Index", 0, n_samples - 1, 0)
        
        # Get this customer's SHAP values
        cust_shap = shap_matrix[customer_idx]
        cust_features = X_test_shap[customer_idx]
        
        # Sort by absolute SHAP value
        sort_idx = np.argsort(np.abs(cust_shap))[::-1][:15]
        
        # Waterfall-style chart
        colors = ['#f5576c' if v > 0 else '#43e97b' for v in cust_shap[sort_idx]]
        
        fig = go.Figure(go.Bar(
            x=cust_shap[sort_idx][::-1],
            y=[feat_names[i] for i in sort_idx][::-1],
            orientation='h',
            marker_color=colors[::-1],
        ))
        fig.update_layout(
            title=f'Customer #{customer_idx} — Feature Contributions to Prediction',
            height=500, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=16,
            xaxis_title='SHAP Value (→ churn / ← retain)',
        )
        fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary for this customer
        churn_drivers = [(feat_names[i], cust_shap[i]) for i in sort_idx if cust_shap[i] > 0][:5]
        retain_drivers = [(feat_names[i], cust_shap[i]) for i in sort_idx if cust_shap[i] < 0][:5]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔴 Pushing Toward Churn")
            if churn_drivers:
                for feat, val in churn_drivers:
                    st.markdown(f"- **{feat}** (+{val:.4f})")
            else:
                st.markdown("_No significant churn drivers_")
        with col2:
            st.markdown("##### 🟢 Pushing Toward Retention")
            if retain_drivers:
                for feat, val in retain_drivers:
                    st.markdown(f"- **{feat}** ({val:.4f})")
            else:
                st.markdown("_No significant retention drivers_")
    
    # ─── Tab 3: Feature Interactions ───
    with tab3:
        st.markdown("#### 🔗 SHAP Feature Dependence")
        st.markdown("How does a feature's value relate to its SHAP impact?")
        
        # Top features for selection
        top_feat_idx = np.argsort(mean_abs)[::-1][:15]
        top_feat_names = [feat_names[i] for i in top_feat_idx]
        
        selected_feat = st.selectbox("Select Feature", top_feat_names)
        feat_idx_sel = feat_names.index(selected_feat)
        
        # SHAP dependence plot
        fig = px.scatter(
            x=X_test_shap[:, feat_idx_sel],
            y=shap_matrix[:, feat_idx_sel],
            color=shap_matrix[:, feat_idx_sel],
            color_continuous_scale='RdBu_r',
            labels={'x': f'{selected_feat} (feature value)', 'y': 'SHAP Value', 'color': 'SHAP'},
            title=f'SHAP Dependence — {selected_feat}',
            opacity=0.6,
        )
        fig.update_layout(
            height=450, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'), title_font_size=16,
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <b>📖 How to read this:</b><br>
            • Each dot is a customer<br>
            • X-axis = the actual feature value<br>
            • Y-axis = SHAP value (above 0 → pushes toward churn)<br>
            • This shows the <b>non-linear relationship</b> between feature value and churn impact
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 6: RETENTION INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "💡 Retention Insights":
    st.markdown('<p class="section-header">💡 Actionable Retention Insights</p>', unsafe_allow_html=True)
    st.markdown("Data-backed strategies to reduce customer churn — derived from SHAP model analysis")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if shap_data is None:
        st.warning("⚠️ SHAP values not computed. Showing general insights from data analysis.")
        use_shap_insights = False
    else:
        use_shap_insights = True
    
    # ─── Executive Summary ───
    total = len(df)
    churned = df['Churn'].value_counts().get('Yes', 0)
    churn_rate = churned / total * 100
    monthly_revenue_at_risk = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    
    st.markdown("### 📊 Executive Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("Customers at Risk", f"{churned:,}", "metric-card metric-card-red"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Monthly Revenue at Risk", f"${monthly_revenue_at_risk:,.0f}", "metric-card metric-card-orange"), unsafe_allow_html=True)
    with col3:
        avg_churn_charge = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
        st.markdown(metric_card("Avg Churner Bill", f"${avg_churn_charge:.0f}/mo", "metric-card metric-card-blue"), unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── SHAP-Backed Insights ───
    if use_shap_insights:
        st.markdown("### 🧠 SHAP-Backed Retention Strategies")
        st.markdown("_Ranked by actual model feature importance — these are the factors the ML model identified as most impactful._")
        
        sys.path.insert(0, BASE_DIR)
        from src.shap_analysis import generate_retention_insights, get_top_churn_drivers
        
        shap_vals = shap_data['shap_values']
        feat_names = shap_data['feature_names']
        X_test_shap = shap_data['X_test']
        
        insights = generate_retention_insights(shap_vals, feat_names, X_test_shap)
        
        if insights:
            for i, insight in enumerate(insights):
                impact_color = {'HIGH': '#ff416c', 'MEDIUM': '#f7971e', 'LOW': '#43e97b'}.get(insight['impact'], '#667eea')
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                    border-radius: 12px;
                    padding: 20px;
                    border-left: 5px solid {impact_color};
                    margin: 12px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #333;">{insight['title']}</h4>
                        <span style="
                            background: {impact_color};
                            color: white;
                            padding: 4px 12px;
                            border-radius: 20px;
                            font-size: 12px;
                            font-weight: 600;
                        ">{insight['impact']} IMPACT</span>
                    </div>
                    <p style="color: #555; margin: 8px 0;">{insight['description']}</p>
                    <p style="color: #333; margin: 4px 0;"><b>🎯 Recommended Action:</b> {insight['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific insights could be generated. Check SHAP computation.")
    
    # ─── Data-Backed Segment Analysis ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 High-Risk Customer Segments")
    
    # Contract analysis
    col1, col2 = st.columns(2)
    
    with col1:
        contract_churn = df.groupby('Contract').agg(
            Total=('Churn', 'count'),
            Churned=('Churn', lambda x: (x == 'Yes').sum()),
        ).reset_index()
        contract_churn['Churn Rate'] = (contract_churn['Churned'] / contract_churn['Total'] * 100).round(1)
        contract_churn['Revenue at Risk'] = df[df['Churn'] == 'Yes'].groupby('Contract')['MonthlyCharges'].sum().values
        
        st.markdown("#### By Contract Type")
        st.dataframe(contract_churn, use_container_width=True, hide_index=True)
    
    with col2:
        # Internet service analysis
        internet_churn = df.groupby('InternetService').agg(
            Total=('Churn', 'count'),
            Churned=('Churn', lambda x: (x == 'Yes').sum()),
        ).reset_index()
        internet_churn['Churn Rate'] = (internet_churn['Churned'] / internet_churn['Total'] * 100).round(1)
        
        st.markdown("#### By Internet Service")
        st.dataframe(internet_churn, use_container_width=True, hide_index=True)
    
    # ─── Retention Playbook ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Retention Playbook — Priority Actions")
    
    playbook = [
        ("1️⃣", "Convert Month-to-Month", "Offer 15-20% discount for switching to annual contracts", "Reduces churn by ~30%", "HIGH"),
        ("2️⃣", "New Customer Onboarding", "Implement a 90-day welcome program with check-ins at day 7, 30, 60, 90", "Targets highest-risk segment (0-12 months)", "HIGH"),
        ("3️⃣", "Service Bundling Push", "Cross-sell security, backup, and support to single-service customers", "Each additional service reduces churn probability", "HIGH"),
        ("4️⃣", "Fiber Optic Satisfaction", "Audit fiber performance; offer speed/SLA guarantees", "Fiber churn rate 42% vs DSL 19%", "MEDIUM"),
        ("5️⃣", "Auto-Pay Migration", "Give $5/mo credit for switching from electronic check to auto-pay", "E-check users churn at 45% vs 15% for auto-pay", "MEDIUM"),
        ("6️⃣", "Senior Citizen Program", "Dedicated support line + simplified billing for seniors", "Senior churn 41% vs non-senior 24%", "MEDIUM"),
        ("7️⃣", "Win-Back Campaign", "Targeted offers for customers predicted >60% churn probability", "Use model predictions for proactive outreach", "HIGH"),
    ]
    
    for icon, title, action, rationale, impact in playbook:
        impact_color = '#ff416c' if impact == 'HIGH' else '#f7971e'
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 10px;
            padding: 16px 20px;
            margin: 8px 0;
            border-left: 4px solid {impact_color};
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            display: flex;
            gap: 16px;
            align-items: flex-start;
        ">
            <span style="font-size: 28px;">{icon}</span>
            <div>
                <b style="font-size: 16px; color: #333;">{title}</b>
                <p style="color: #555; margin: 4px 0 2px 0;">{action}</p>
                <p style="color: #888; font-size: 13px; margin: 0;"><i>📊 {rationale}</i></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
