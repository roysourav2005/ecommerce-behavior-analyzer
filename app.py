# app.py — Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import build_master_df, train_model, get_summary_stats

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="E-Commerce Behavior Analyzer",
    page_icon="🛒",
    layout="wide"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        }
        h1 { color: #1a1a2e; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────

@st.cache_data
def load():
    df    = build_master_df()
    stats = get_summary_stats(df)
    model, acc, report, importances, X_test, y_test, y_pred = train_model(df)
    return df, stats, model, acc, report, importances

with st.spinner("Loading data and training model... please wait ⏳"):
    df, stats, model, acc, report, importances = load()

# ─── HEADER ──────────────────────────────────────────────────────────────────

st.title("🛒 E-Commerce Customer Behavior Analyzer")
st.markdown("**Dataset:** Olist Brazilian E-Commerce &nbsp;|&nbsp; **Model:** Random Forest Classifier")
st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚚 Delivery Analysis",
    "⭐ Review Analysis",
    "🤖 ML Model"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📦 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",       f"{stats['total_orders']:,}")
    col2.metric("Delayed Orders",     f"{stats['delayed_orders']:,}")
    col3.metric("Delay Rate",         f"{stats['delay_rate']}%")
    col4.metric("Avg Review Score",   f"{stats['avg_review_score']} ⭐")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Order Value",    f"R$ {stats['avg_order_value']}")
    col6.metric("Avg Items/Order",    f"{stats['avg_items_per_order']}")
    col7.metric("On-Time Orders",     f"{stats['on_time_orders']:,}")
    col8.metric("Avg Delay (days)",   f"{stats['avg_delay_days']} days")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🗺️ Orders by State")
        state_counts = df['customer_state'].value_counts().reset_index()
        state_counts.columns = ['state', 'orders']
        fig = px.bar(
            state_counts.head(15),
            x='state', y='orders',
            color='orders',
            color_continuous_scale='Blues',
            title="Top 15 States by Order Volume"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("💳 Payment Methods")
        pay_counts = df['payment_type'].value_counts().reset_index()
        pay_counts.columns = ['payment_type', 'count']
        fig2 = px.pie(
            pay_counts,
            names='payment_type',
            values='count',
            title="Payment Type Distribution",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Orders Over Time")
    df['purchase_month_year'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    orders_time = df.groupby('purchase_month_year').size().reset_index(name='orders')
    fig3 = px.line(
        orders_time,
        x='purchase_month_year', y='orders',
        title="Monthly Order Volume",
        markers=True,
        color_discrete_sequence=['#0077b6']
    )
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DELIVERY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🚚 Delivery Delay Analysis")

    col1, col2 = st.columns(2)

    with col1:
        delay_dist = df['is_delayed'].value_counts().reset_index()
        delay_dist.columns = ['status', 'count']
        delay_dist['status'] = delay_dist['status'].map({0: 'On Time ✅', 1: 'Delayed ❌'})
        fig = px.pie(
            delay_dist,
            names='status', values='count',
            title="On-Time vs Delayed Orders",
            color_discrete_sequence=['#0077b6', '#e63946']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df[df['delivery_delay_days'].between(-30, 60)],
            x='delivery_delay_days',
            nbins=60,
            title="Delivery Delay Distribution (days)",
            color_discrete_sequence=['#0077b6'],
            labels={'delivery_delay_days': 'Delay Days (negative = early)'}
        )
        fig2.add_vline(x=0, line_dash="dash", line_color="red",
                       annotation_text="On-time boundary")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📅 Delay by Day of Week")
    dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    df['day_name'] = df['purchase_dayofweek'].map(dow_map)
    delay_dow = df.groupby('day_name')['is_delayed'].mean().reset_index()
    delay_dow.columns = ['day', 'delay_rate']
    delay_dow['delay_rate'] = (delay_dow['delay_rate'] * 100).round(2)
    day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    delay_dow['day'] = pd.Categorical(delay_dow['day'], categories=day_order, ordered=True)
    delay_dow = delay_dow.sort_values('day')
    fig3 = px.bar(
        delay_dow, x='day', y='delay_rate',
        title="Delay Rate by Purchase Day of Week (%)",
        color='delay_rate',
        color_continuous_scale='Reds',
        labels={'delay_rate': 'Delay Rate (%)'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🗺️ Delay Rate by State")
    state_delay = df.groupby('customer_state')['is_delayed'].mean().reset_index()
    state_delay.columns = ['state', 'delay_rate']
    state_delay['delay_rate'] = (state_delay['delay_rate'] * 100).round(2)
    state_delay = state_delay.sort_values('delay_rate', ascending=False)
    fig4 = px.bar(
        state_delay,
        x='state', y='delay_rate',
        title="Delay Rate by Customer State (%)",
        color='delay_rate',
        color_continuous_scale='Oranges',
        labels={'delay_rate': 'Delay Rate (%)'}
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — REVIEW ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("⭐ Customer Review Analysis")

    col1, col2 = st.columns(2)

    with col1:
        review_dist = df['review_score'].value_counts().sort_index().reset_index()
        review_dist.columns = ['score', 'count']
        fig = px.bar(
            review_dist,
            x='score', y='count',
            title="Review Score Distribution",
            color='score',
            color_continuous_scale='RdYlGn',
            labels={'score': 'Review Score', 'count': 'Number of Orders'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        review_delay = df.groupby('is_delayed')['review_score'].mean().reset_index()
        review_delay['status'] = review_delay['is_delayed'].map(
            {0: 'On Time ✅', 1: 'Delayed ❌'}
        )
        fig2 = px.bar(
            review_delay,
            x='status', y='review_score',
            title="Avg Review Score: On-Time vs Delayed",
            color='status',
            color_discrete_map={'On Time ✅': '#0077b6', 'Delayed ❌': '#e63946'},
            labels={'review_score': 'Avg Review Score'}
        )
        fig2.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("💰 Order Value vs Review Score")
    fig3 = px.box(
        df[df['total_price'] < 1000],
        x='review_score', y='total_price',
        title="Order Value Distribution by Review Score",
        color='review_score',
        color_discrete_sequence=px.colors.sequential.Blues,
        labels={'total_price': 'Order Value (R$)', 'review_score': 'Review Score'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML MODEL
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("🤖 Delivery Delay Prediction Model")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model",    "Random Forest")
    col2.metric("Accuracy", f"{round(acc * 100, 2)}%")
    col3.metric("Features", "13")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🔍 Feature Importance")
        imp_df = importances.reset_index()
        imp_df.columns = ['feature', 'importance']
        fig = px.bar(
            imp_df,
            x='importance', y='feature',
            orientation='h',
            title="Which features predict delay the most?",
            color='importance',
            color_continuous_scale='Blues',
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📊 Model Performance")
        report_df = pd.DataFrame(report).transpose().round(2)
        report_df = report_df[report_df.index.isin(['0', '1'])]
        report_df.index = ['On Time ✅', 'Delayed ❌']
        st.dataframe(report_df[['precision','recall','f1-score','support']], use_container_width=True)

        st.markdown("---")
        st.subheader("🔮 Predict a Single Order")
        st.markdown("Adjust the sliders and see if the order will be delayed:")

        c1, c2 = st.columns(2)
        with c1:
            item_count   = st.slider("Number of Items",      1, 20, 2)
            total_price  = st.slider("Total Price (R$)",     10, 5000, 150)
            total_freight= st.slider("Freight Value (R$)",   5, 200, 20)
            processing_t = st.slider("Processing Time (hrs)",0, 72, 10)
            shipping_t   = st.slider("Shipping Time (hrs)",  0, 500, 100)
        with c2:
            installments = st.slider("Payment Installments", 1, 24, 1)
            pay_type_enc = st.selectbox("Payment Type", [0,1,2,3],
                          format_func=lambda x: ['credit_card','boleto','voucher','debit_card'][x])
            state_enc    = st.slider("Customer State (encoded)", 0, 26, 10)
            hour         = st.slider("Purchase Hour",        0, 23, 14)
            dow          = st.slider("Day of Week (0=Mon)",  0, 6, 2)
            month        = st.slider("Month",                1, 12, 6)

        input_data = pd.DataFrame([{
            'item_count'            : item_count,
            'total_price'           : total_price,
            'total_freight'         : total_freight,
            'avg_price'             : total_price / item_count,
            'total_payment'         : total_price + total_freight,
            'payment_installments'  : installments,
            'payment_type_enc'      : pay_type_enc,
            'state_enc'             : state_enc,
            'purchase_hour'         : hour,
            'purchase_dayofweek'    : dow,
            'purchase_month'        : month,
            'processing_time'       : processing_t,
            'shipping_time'         : shipping_t,
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        if pred == 1:
            st.error(f"❌ Likely DELAYED — Delay probability: {round(prob[1]*100, 1)}%")
        else:
            st.success(f"✅ Likely ON TIME — On-time probability: {round(prob[0]*100, 1)}%")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with ❤️ using Python · Pandas · Scikit-learn · Plotly · Streamlit")