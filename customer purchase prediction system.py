import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📦 Customer Retention Prediction Dashboard")
st.markdown("Predict **repeat purchase probability within 30 days** for first-time customers")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Online Retail CSV file", type=["csv"]
)

# ---------------- TRAIN MODEL FUNCTION ----------------
@st.cache_resource
def train_model(df):
    # Cleaning
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    df['InvoiceDate'] = pd.to_datetime(
        df['InvoiceDate'],
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['InvoiceDate'])

    # rest of your feature engineering & model code

    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # First purchase
    first_purchase = (
        df.sort_values('InvoiceDate')
          .groupby('CustomerID')
          .first()
          .reset_index()
    )

    # Target creation (30 days repeat)
    df_merge = df.merge(
        first_purchase[['CustomerID', 'InvoiceDate']],
        on='CustomerID',
        suffixes=('', '_first')
    )

    df_merge['days_diff'] = (
        df_merge['InvoiceDate'] - df_merge['InvoiceDate_first']
    ).dt.days

    repeat_customers = df_merge[
        (df_merge['days_diff'] > 0) & (df_merge['days_diff'] <= 30)
    ]['CustomerID'].unique()

    first_purchase['repeat_30d'] = first_purchase['CustomerID']\
        .isin(repeat_customers).astype(int)

    # Feature engineering
    fp = first_purchase.copy()
    fp['purchase_hour'] = fp['InvoiceDate'].dt.hour
    fp['purchase_day'] = fp['InvoiceDate'].dt.dayofweek
    fp['num_items'] = fp['Quantity']
    fp['avg_item_price'] = fp['UnitPrice']
    fp['total_amount'] = fp['TotalAmount']
    fp['is_bulk'] = (fp['Quantity'] >= 10).astype(int)

    X = fp[['purchase_hour','purchase_day','num_items',
            'avg_item_price','total_amount','is_bulk']]
    y = fp['repeat_30d']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_scaled, y)

    return fp, model, scaler

# ---------------- MAIN APP ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    fp, model, scaler = train_model(df)

    X_pred = fp[['purchase_hour','purchase_day','num_items',
                 'avg_item_price','total_amount','is_bulk']]
    X_scaled = scaler.transform(X_pred)

    fp['repeat_probability'] = model.predict_proba(X_scaled)[:, 1]

    fp['action'] = pd.cut(
        fp['repeat_probability'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Send Discount', 'Send Recommendations', 'No Action']
    )

    # ---------------- KPI METRICS ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(fp))
    col2.metric("High Risk Customers", len(fp[fp['repeat_probability'] < 0.4]))
    col3.metric("Avg Retention Probability", round(fp['repeat_probability'].mean(), 2))

    st.divider()

    # ---------------- DISTRIBUTION ----------------
    st.subheader("📈 Retention Probability Distribution")
    fig1 = px.histogram(fp, x="repeat_probability", nbins=30)
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- ACTION PIE ----------------
    st.subheader("🎯 Recommended Actions")
    fig2 = px.pie(fp, names="action")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TABLE ----------------
    st.subheader("🧾 Customer-Level Predictions")
    filter_action = st.selectbox(
        "Filter by Action",
        ['All', 'Send Discount', 'Send Recommendations', 'No Action']
    )

    table = fp[['CustomerID','repeat_probability','action']]\
        .sort_values('repeat_probability')

    if filter_action != 'All':
        table = table[table['action'] == filter_action]

    st.dataframe(table)

    # ---------------- DOWNLOAD ----------------
    st.subheader("⬇️ Download Marketing Target List")
    csv = table.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="customer_retention_targets.csv",
        mime="text/csv"
    )

    # ---------------- REAL-TIME PREDICTION ----------------
    st.divider()
    st.subheader("⚡ Real-Time Customer Prediction")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            purchase_hour = st.slider("Purchase Hour", 0, 23, 12)
            purchase_day = st.slider("Purchase Day (0=Mon)", 0, 6, 2)

        with c2:
            num_items = st.number_input("Number of Items", 1, 100, 3)
            avg_item_price = st.number_input("Avg Item Price", 1.0, 500.0, 20.0)

        with c3:
            total_amount = st.number_input("Total Amount", 1.0, 5000.0, 120.0)
            is_bulk = st.selectbox("Bulk Order?", [0, 1])

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = np.array([[
            purchase_hour, purchase_day, num_items,
            avg_item_price, total_amount, is_bulk
        ]])

        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        if prob < 0.4:
            action = "🎁 Send Discount"
        elif prob < 0.7:
            action = "📩 Send Recommendations"
        else:
            action = "✅ No Action Needed"

        st.success(f"Repeat Purchase Probability: {prob:.2f}")
        st.info(f"Recommended Action: {action}")

else:
    st.info("👈 Upload the Online Retail CSV file to start")
