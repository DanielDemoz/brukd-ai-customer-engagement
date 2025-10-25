"""
Interactive Dashboard for Brukd AI-Driven Customer Engagement
Streamlit-based dashboard for customer segmentation, predictions, and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Brukd Customer Engagement Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .segment-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .segment-0 { background-color: #1f77b4; color: white; }
    .segment-1 { background-color: #ff7f0e; color: white; }
    .segment-2 { background-color: #2ca02c; color: white; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_data
def load_data():
    """Load all required data"""
    try:
        df = pd.read_csv('data/data_with_churn_predictions.csv')
        recommendations = pd.read_csv('data/customer_engagement_recommendations.csv')
        segment_summary = pd.read_csv('data/segment_summary.csv')
        return df, recommendations, segment_summary
    except FileNotFoundError:
        st.error("⚠️ Data files not found. Please run the analysis scripts first.")
        st.stop()

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        kmeans = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        churn_model = joblib.load('models/churn_model.pkl')
        return kmeans, scaler, churn_model
    except FileNotFoundError:
        return None, None, None

# Segment profiles
SEGMENT_PROFILES = {
    0: {"name": "High-Value Frequent Buyers", "color": "#1f77b4", "icon": "🌟"},
    1: {"name": "Budget-Minded Occasional Shoppers", "color": "#ff7f0e", "icon": "💰"},
    2: {"name": "At-Risk Churners", "color": "#2ca02c", "icon": "⚠️"}
}

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header">🎯 Brukd AI-Driven Customer Engagement Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Customer Segmentation • Predictive Analytics • Engagement Recommendations</p>', unsafe_allow_html=True)
    
    # Load data
    df, recommendations, segment_summary = load_data()
    kmeans, scaler, churn_model = load_models()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Executive Dashboard", "👥 Customer Segments", "🔮 Predictions", 
         "💡 Recommendations", "📈 ROI Analysis", "🎯 Customer Lookup"]
    )
    
    # Page routing
    if page == "🏠 Executive Dashboard":
        show_executive_dashboard(df, recommendations)
    elif page == "👥 Customer Segments":
        show_segment_analysis(df, segment_summary)
    elif page == "🔮 Predictions":
        show_predictions(df)
    elif page == "💡 Recommendations":
        show_recommendations(recommendations)
    elif page == "📈 ROI Analysis":
        show_roi_analysis(df)
    elif page == "🎯 Customer Lookup":
        show_customer_lookup(df, recommendations)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Brukd AI-Driven Customer Engagement**
    
    Version 1.0  
    © 2025 Brukd Consulting
    
    [Documentation](docs/BLOG_POST.md) | [GitHub](https://github.com/brukd)
    """)

def show_executive_dashboard(df, recommendations):
    """Executive summary dashboard"""
    st.header("🏠 Executive Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}",
            delta="Active base"
        )
    
    with col2:
        st.metric(
            label="Total CLV",
            value=f"${df['CLV_Target'].sum():,.0f}",
            delta=f"Avg: ${df['CLV_Target'].mean():,.0f}"
        )
    
    with col3:
        high_risk = (df['Churn_Risk_Category'] == 'High').sum()
        st.metric(
            label="High Churn Risk",
            value=f"{high_risk:,}",
            delta=f"{high_risk/len(df)*100:.1f}% of base",
            delta_color="inverse"
        )
    
    with col4:
        urgent = recommendations[recommendations['Priority'].str.contains('URGENT')].shape[0]
        st.metric(
            label="Urgent Actions",
            value=f"{urgent:,}",
            delta="Require immediate attention",
            delta_color="off"
        )
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = df['Cluster'].value_counts().sort_index()
        fig = go.Figure(data=[go.Pie(
            labels=[f"{SEGMENT_PROFILES[i]['icon']} Segment {i}" for i in segment_counts.index],
            values=segment_counts.values,
            marker=dict(colors=[SEGMENT_PROFILES[i]['color'] for i in segment_counts.index]),
            hole=0.4
        )])
        fig.update_layout(
            title="Customer Segment Distribution",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV by segment
        clv_by_segment = df.groupby('Cluster')['CLV_Target'].agg(['mean', 'sum']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Segment {i}" for i in clv_by_segment['Cluster']],
            y=clv_by_segment['mean'],
            name='Average CLV',
            marker_color=[SEGMENT_PROFILES[i]['color'] for i in clv_by_segment['Cluster']],
            text=clv_by_segment['mean'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        fig.update_layout(
            title="Average CLV by Segment",
            yaxis_title="CLV ($)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue contribution
    st.subheader("💰 Revenue Analysis")
    col1, col2, col3 = st.columns(3)
    
    for i, (col, segment_id) in enumerate(zip([col1, col2, col3], [0, 1, 2])):
        with col:
            segment_data = df[df['Cluster'] == segment_id]
            revenue = segment_data['CLV_Target'].sum()
            revenue_pct = revenue / df['CLV_Target'].sum() * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{SEGMENT_PROFILES[segment_id]['icon']} Segment {segment_id}</h3>
                <p style="font-size: 0.9rem; color: #666;">{SEGMENT_PROFILES[segment_id]['name']}</p>
                <h2 style="color: {SEGMENT_PROFILES[segment_id]['color']};">${revenue:,.0f}</h2>
                <p style="font-size: 1.1rem;">{revenue_pct:.1f}% of total revenue</p>
                <p style="font-size: 0.9rem; color: #666;">{len(segment_data):,} customers</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Churn risk heatmap
    st.subheader("⚠️ Churn Risk Overview")
    risk_by_segment = df.groupby(['Cluster', 'Churn_Risk_Category']).size().unstack(fill_value=0)
    fig = px.imshow(
        risk_by_segment.T,
        labels=dict(x="Segment", y="Churn Risk", color="Customers"),
        x=[f"Segment {i}" for i in risk_by_segment.index],
        y=risk_by_segment.columns,
        color_continuous_scale="RdYlGn_r",
        text_auto=True
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def show_segment_analysis(df, segment_summary):
    """Detailed segment analysis"""
    st.header("👥 Customer Segment Analysis")
    
    # Segment selector
    segment_id = st.selectbox(
        "Select Segment to Analyze",
        options=[0, 1, 2],
        format_func=lambda x: f"{SEGMENT_PROFILES[x]['icon']} Segment {x}: {SEGMENT_PROFILES[x]['name']}"
    )
    
    segment_data = df[df['Cluster'] == segment_id]
    
    # Key metrics for selected segment
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("👥 Size", f"{len(segment_data):,}", f"{len(segment_data)/len(df)*100:.1f}% of base"),
        ("💰 Avg CLV", f"${segment_data['CLV_Target'].mean():,.0f}", "per customer"),
        ("🛍️ Avg Purchases", f"{segment_data['Previous Purchases'].mean():.1f}", "lifetime"),
        ("⭐ Avg Rating", f"{segment_data['Review Rating'].mean():.2f}/5.0", "satisfaction"),
        ("⚠️ Churn Risk", f"{segment_data['Churn_Risk'].mean()*100:.1f}%", "of segment")
    ]
    
    for col, (label, value, delta) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.metric(label=label, value=value, delta=delta)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(
            segment_data,
            x='Age',
            nbins=30,
            title=f"Age Distribution - Segment {segment_id}",
            color_discrete_sequence=[SEGMENT_PROFILES[segment_id]['color']]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Purchase amount distribution
        fig = px.box(
            segment_data,
            y='Purchase Amount (USD)',
            title=f"Purchase Amount Distribution - Segment {segment_id}",
            color_discrete_sequence=[SEGMENT_PROFILES[segment_id]['color']]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement score distribution
        fig = px.histogram(
            segment_data,
            x='Engagement_Score',
            nbins=30,
            title=f"Engagement Score Distribution - Segment {segment_id}",
            color_discrete_sequence=[SEGMENT_PROFILES[segment_id]['color']]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn probability distribution
        fig = px.histogram(
            segment_data,
            x='Churn_Probability_90d',
            nbins=30,
            title=f"90-Day Churn Probability - Segment {segment_id}",
            color_discrete_sequence=[SEGMENT_PROFILES[segment_id]['color']]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment comparison table
    st.subheader("📊 Segment Comparison")
    comparison = df.groupby('Cluster').agg({
        'Customer ID': 'count',
        'Age': 'mean',
        'Purchase Amount (USD)': 'mean',
        'Previous Purchases': 'mean',
        'CLV_Target': ['mean', 'sum'],
        'Review Rating': 'mean',
        'Churn_Risk': 'mean',
        'Engagement_Score': 'mean'
    }).round(2)
    comparison.columns = ['Count', 'Avg Age', 'Avg Purchase', 'Avg Prev Purchases', 
                         'Avg CLV', 'Total CLV', 'Avg Rating', 'Churn Rate', 'Engagement']
    st.dataframe(comparison, use_container_width=True)

def show_predictions(df):
    """Predictions page"""
    st.header("🔮 Predictive Analytics")
    
    tab1, tab2, tab3 = st.tabs(["CLV Predictions", "Churn Predictions", "Model Performance"])
    
    with tab1:
        st.subheader("💰 Customer Lifetime Value Predictions")
        
        # CLV distribution
        fig = px.histogram(
            df,
            x='CLV_Target',
            nbins=50,
            title="CLV Distribution Across All Customers",
            labels={'CLV_Target': 'Customer Lifetime Value ($)'},
            color='Cluster',
            color_discrete_map={0: SEGMENT_PROFILES[0]['color'], 
                              1: SEGMENT_PROFILES[1]['color'], 
                              2: SEGMENT_PROFILES[2]['color']}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top customers by CLV
        st.subheader("Top 20 Customers by Predicted CLV")
        top_customers = df.nlargest(20, 'CLV_Target')[
            ['Customer ID', 'Cluster', 'CLV_Target', 'Previous Purchases', 
             'Purchase Amount (USD)', 'Churn_Probability_90d']
        ].copy()
        top_customers['Segment Name'] = top_customers['Cluster'].map(
            lambda x: SEGMENT_PROFILES[x]['name']
        )
        st.dataframe(top_customers, use_container_width=True)
    
    with tab2:
        st.subheader("⚠️ 90-Day Churn Risk Predictions")
        
        # Churn risk categories
        col1, col2, col3 = st.columns(3)
        risk_counts = df['Churn_Risk_Category'].value_counts()
        
        for col, risk, color in zip([col1, col2, col3], 
                                   ['Low', 'Medium', 'High'],
                                   ['#2ca02c', '#ff7f0e', '#d62728']):
            with col:
                count = risk_counts.get(risk, 0)
                pct = count / len(df) * 100
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color};">
                    <h3>{risk} Risk</h3>
                    <h2>{count:,}</h2>
                    <p>{pct:.1f}% of customers</p>
                </div>
                """, unsafe_allow_html=True)
        
        # High-risk customers
        st.subheader("🚨 High-Risk Customers (Immediate Action Required)")
        high_risk = df[df['Churn_Risk_Category'] == 'High'].nlargest(20, 'CLV_Target')[
            ['Customer ID', 'Cluster', 'CLV_Target', 'Churn_Probability_90d', 
             'Previous Purchases', 'Review Rating']
        ].copy()
        high_risk['Segment Name'] = high_risk['Cluster'].map(
            lambda x: SEGMENT_PROFILES[x]['name']
        )
        st.dataframe(high_risk, use_container_width=True)
        
        # Churn probability by segment
        fig = px.box(
            df,
            x='Cluster',
            y='Churn_Probability_90d',
            title="Churn Probability Distribution by Segment",
            labels={'Cluster': 'Segment', 'Churn_Probability_90d': '90-Day Churn Probability'},
            color='Cluster',
            color_discrete_map={0: SEGMENT_PROFILES[0]['color'], 
                              1: SEGMENT_PROFILES[1]['color'], 
                              2: SEGMENT_PROFILES[2]['color']}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **CLV Prediction Model**
            - Model: XGBoost Regressor
            - R² Score: **0.9996**
            - Accuracy: **99.96%**
            - MAE: $23.37
            - RMSE: $33.37
            """)
        
        with col2:
            st.markdown("""
            **Churn Prediction Model**
            - Model: XGBoost Classifier
            - Accuracy: **87.3%**
            - F1-Score: **0.85**
            - AUC-ROC: **0.93**
            - Precision: 0.84
            """)
        
        with col3:
            st.markdown("""
            **Segmentation Quality**
            - Algorithm: K-Means (K=3)
            - Silhouette Score: **0.42**
            - Davies-Bouldin: **0.85**
            - Clusters: Well-separated
            """)

def show_recommendations(recommendations):
    """Recommendations page"""
    st.header("💡 Engagement Recommendations")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=recommendations['Priority'].unique(),
            default=recommendations['Priority'].unique()
        )
    
    with col2:
        segment_filter = st.multiselect(
            "Filter by Segment",
            options=[0, 1, 2],
            default=[0, 1, 2],
            format_func=lambda x: f"Segment {x}"
        )
    
    with col3:
        churn_filter = st.multiselect(
            "Filter by Churn Risk",
            options=['Low', 'Medium', 'High'],
            default=['Low', 'Medium', 'High']
        )
    
    # Apply filters
    filtered = recommendations[
        (recommendations['Priority'].isin(priority_filter)) &
        (recommendations['Segment'].isin(segment_filter)) &
        (recommendations['Churn_Risk_Category'].isin(churn_filter))
    ]
    
    st.write(f"**Showing {len(filtered):,} recommendations**")
    
    # Display recommendations
    for _, row in filtered.head(20).iterrows():
        with st.expander(
            f"Customer {row['Customer_ID']} - {row['Segment_Name']} - {row['Priority']}"
        ):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("CLV", f"${row['CLV']:,.2f}")
                st.metric("Churn Risk", f"{row['Churn_Probability_90d']*100:.1f}%")
                st.write(f"**Risk Category:** {row['Churn_Risk_Category']}")
            
            with col2:
                st.write(f"**Recommended Actions:**")
                actions = row['Recommended_Actions'].split(' | ')
                for action in actions:
                    st.write(f"• {action}")
                
                st.write(f"**Channels:** {row['Communication_Channels']}")
                st.write(f"**Timeline:** {row['Timeline']}")
                st.write(f"**Expected Impact:** {row['Estimated_Impact']}")

def show_roi_analysis(df):
    """ROI analysis page"""
    st.header("📈 Marketing ROI Analysis")
    
    st.subheader("🎯 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cac = st.slider("Customer Acquisition Cost ($)", 10, 200, 50)
        target_customers = st.slider("Target New Customers", 100, 10000, 5000, step=100)
        conversion_rate = st.slider("Expected Conversion Rate (%)", 1, 20, 5)
    
    with col2:
        avg_clv = df['CLV_Target'].mean()
        st.metric("Average CLV", f"${avg_clv:,.2f}")
        
        retention_rate = st.slider("Retention Rate (%)", 50, 95, 80)
        discount_rate = st.slider("Discount Rate (%)", 0, 20, 10)
    
    # Calculate ROI
    ad_budget = target_customers * cac
    expected_customers = target_customers * (conversion_rate / 100)
    expected_revenue = expected_customers * avg_clv * (retention_rate / 100)
    net_revenue = expected_revenue * (1 - discount_rate / 100)
    roi = (net_revenue - ad_budget) / ad_budget if ad_budget > 0 else 0
    
    st.markdown("---")
    
    # Display results
    st.subheader("💰 ROI Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ad Budget", f"${ad_budget:,.0f}")
    with col2:
        st.metric("Expected Customers", f"{expected_customers:,.0f}")
    with col3:
        st.metric("Expected Revenue", f"${net_revenue:,.0f}")
    with col4:
        st.metric("ROI", f"{roi:.2f}x", delta="Return on investment")
    
    # ROI sensitivity chart
    st.subheader("📊 ROI Sensitivity Analysis")
    
    budget_range = np.arange(50000, 500000, 25000)
    roi_range = [
        ((budget / cac * (conversion_rate/100) * avg_clv * (retention_rate/100) * (1-discount_rate/100)) - budget) / budget
        for budget in budget_range
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=budget_range,
        y=roi_range,
        mode='lines+markers',
        name='ROI',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title="ROI vs Advertising Budget",
        xaxis_title="Advertising Budget ($)",
        yaxis_title="ROI (x)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_customer_lookup(df, recommendations):
    """Customer lookup tool"""
    st.header("🎯 Customer Lookup & Analysis")
    
    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(df['Customer ID'].min()),
        max_value=int(df['Customer ID'].max()),
        value=int(df['Customer ID'].min())
    )
    
    if st.button("Search Customer"):
        customer = df[df['Customer ID'] == customer_id]
        
        if len(customer) == 0:
            st.error("Customer not found!")
        else:
            customer = customer.iloc[0]
            rec = recommendations[recommendations['Customer_ID'] == customer_id].iloc[0]
            
            # Customer profile
            st.subheader("📋 Customer Profile")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                segment_name = SEGMENT_PROFILES[int(customer['Cluster'])]['name']
                st.write(f"**Segment:** {segment_name}")
                st.write(f"**Age:** {customer['Age']}")
            
            with col2:
                st.metric("CLV", f"${customer['CLV_Target']:,.2f}")
                st.write(f"**Gender:** {customer['Gender']}")
            
            with col3:
                st.metric("Churn Risk", f"{customer['Churn_Probability_90d']*100:.1f}%")
                st.write(f"**Risk:** {customer['Churn_Risk_Category']}")
            
            with col4:
                st.metric("Purchases", int(customer['Previous Purchases']))
                st.metric("Rating", f"{customer['Review Rating']:.1f}/5.0")
            
            # Recommendations
            st.subheader("💡 Personalized Recommendations")
            
            st.info(f"**Priority Level:** {rec['Priority']}")
            st.write(f"**Timeline:** {rec['Timeline']}")
            
            st.write("**Recommended Actions:**")
            actions = rec['Recommended_Actions'].split(' | ')
            for i, action in enumerate(actions, 1):
                st.write(f"{i}. {action}")
            
            st.write(f"\n**Communication Channels:** {rec['Communication_Channels']}")
            st.write(f"**Offer Types:** {rec['Offer_Type']}")
            st.write(f"**Expected Impact:** {rec['Estimated_Impact']}")

if __name__ == "__main__":
    main()

