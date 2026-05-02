# model.py — Data loading, cleaning, feature engineering, ML model

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD DATA ───────────────────────────────────────────────────────────────

def load_data():
    orders       = pd.read_csv('data/olist_orders_dataset.csv')
    items        = pd.read_csv('data/olist_order_items_dataset.csv')
    customers    = pd.read_csv('data/olist_customers_dataset.csv')
    payments     = pd.read_csv('data/olist_order_payments_dataset.csv')
    reviews      = pd.read_csv('data/olist_order_reviews_dataset.csv')
    products     = pd.read_csv('data/olist_products_dataset.csv')
    translation  = pd.read_csv('data/product_category_name_translation.csv')

    # Merge category names in English
    products = products.merge(translation, on='product_category_name', how='left')
    return orders, items, customers, payments, reviews, products


# ─── CLEAN & MERGE ───────────────────────────────────────────────────────────

def build_master_df():
    orders, items, customers, payments, reviews, products = load_data()

    # Parse datetime columns
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')

    # Keep only delivered orders for delay analysis
    delivered = orders[orders['order_status'] == 'delivered'].copy()

    # Calculate delivery delay in days (positive = late, negative = early)
    delivered['delivery_delay_days'] = (
        delivered['order_delivered_customer_date'] -
        delivered['order_estimated_delivery_date']
    ).dt.days

    # Binary target: 1 = delayed, 0 = on time or early
    delivered['is_delayed'] = (delivered['delivery_delay_days'] > 0).astype(int)

    # Calculate processing time (purchase → approved)
    delivered['processing_time'] = (
        delivered['order_approved_at'] -
        delivered['order_purchase_timestamp']
    ).dt.total_seconds() / 3600  # in hours

    # Calculate shipping time (approved → carrier)
    delivered['shipping_time'] = (
        delivered['order_delivered_carrier_date'] -
        delivered['order_approved_at']
    ).dt.total_seconds() / 3600

    # Extract purchase hour and day of week
    delivered['purchase_hour']    = delivered['order_purchase_timestamp'].dt.hour
    delivered['purchase_dayofweek'] = delivered['order_purchase_timestamp'].dt.dayofweek
    delivered['purchase_month']   = delivered['order_purchase_timestamp'].dt.month

    # Aggregate items per order
    items_agg = items.groupby('order_id').agg(
        item_count       = ('order_item_id', 'count'),
        total_price      = ('price', 'sum'),
        total_freight    = ('freight_value', 'sum'),
        avg_price        = ('price', 'mean'),
    ).reset_index()

    # Aggregate payments
    pay_agg = payments.groupby('order_id').agg(
        total_payment    = ('payment_value', 'sum'),
        payment_installments = ('payment_installments', 'max'),
        payment_type     = ('payment_type', 'first'),
    ).reset_index()

    # Review scores (first review per order)
    review_agg = reviews.groupby('order_id').agg(
        review_score = ('review_score', 'mean')
    ).reset_index()

    # Merge everything
    df = delivered.merge(customers,   on='customer_id',  how='left')
    df = df.merge(items_agg,          on='order_id',     how='left')
    df = df.merge(pay_agg,            on='order_id',     how='left')
    df = df.merge(review_agg,         on='order_id',     how='left')

    # Encode payment type
    le = LabelEncoder()
    df['payment_type_enc'] = le.fit_transform(df['payment_type'].fillna('unknown'))

    # Encode customer state
    df['state_enc'] = le.fit_transform(df['customer_state'].fillna('unknown'))

    return df


# ─── TRAIN MODEL ─────────────────────────────────────────────────────────────

def train_model(df):
    features = [
        'item_count', 'total_price', 'total_freight',
        'avg_price', 'total_payment', 'payment_installments',
        'payment_type_enc', 'state_enc',
        'purchase_hour', 'purchase_dayofweek', 'purchase_month',
        'processing_time', 'shipping_time'
    ]

    df_model = df[features + ['is_delayed']].dropna()

    X = df_model[features]
    y = df_model['is_delayed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importances
    importances = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    return model, acc, report, importances, X_test, y_test, y_pred


# ─── SUMMARY STATS (for dashboard) ──────────────────────────────────────────

def get_summary_stats(df):
    stats = {
        'total_orders'       : len(df),
        'delayed_orders'     : int(df['is_delayed'].sum()),
        'on_time_orders'     : int((df['is_delayed'] == 0).sum()),
        'delay_rate'         : round(df['is_delayed'].mean() * 100, 2),
        'avg_delay_days'     : round(df['delivery_delay_days'].mean(), 2),
        'avg_review_score'   : round(df['review_score'].mean(), 2),
        'avg_order_value'    : round(df['total_price'].mean(), 2),
        'avg_items_per_order': round(df['item_count'].mean(), 2),
    }
    return stats