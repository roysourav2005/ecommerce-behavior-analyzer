# model.py — Data loading from web, cleaning, feature engineering, ML model

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── DATA URLS (Kaggle Olist dataset via GitHub mirror) ──────────────────────

BASE_URL = "https://raw.githubusercontent.com/roysourav2005/ecommerce-behavior-analyzer/main/data/"

URLS = {
    'orders'      : BASE_URL + 'olist_orders_dataset.csv',
    'items'       : BASE_URL + 'olist_order_items_dataset.csv',
    'customers'   : BASE_URL + 'olist_customers_dataset.csv',
    'payments'    : BASE_URL + 'olist_order_payments_dataset.csv',
    'reviews'     : BASE_URL + 'olist_order_reviews_dataset.csv',
    'products'    : BASE_URL + 'olist_products_dataset.csv',
    'translation' : BASE_URL + 'product_category_name_translation.csv',
}