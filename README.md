# 🛒 E-Commerce Customer Behavior Analyzer

An end-to-end data analysis and machine learning project built on 100K+ real Brazilian e-commerce records from Olist.

## 🔴 Live Demo
👉 [Click here to open the dashboard](https://ecommerce-behavior-analyzer-m6nymhhnx6wc75rrshjaf3.streamlit.app/)

## 📊 Features
- **Overview Dashboard** — Orders by state, payment methods, monthly trends
- **Delivery Analysis** — Delay distribution, delay by state and day of week
- **Review Analysis** — Review score distribution, impact of delays on ratings
- **ML Model** — Random Forest classifier to predict delivery delays with real-time single order prediction

## 🛠️ Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization** — Plotly, Seaborn, Matplotlib
- **Dashboard** — Streamlit
- **ML Model** — Random Forest Classifier
- **Deployment** — Streamlit Cloud
- **Version Control** — Git, GitHub

## 📁 Dataset
[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K+ orders from 2016-2018

## 🚀 Run Locally
```bash
git clone https://github.com/roysourav2005/ecommerce-behavior-analyzer.git
cd ecommerce-behavior-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Key Findings
- Overall delivery delay rate across all orders
- Shipping time is the strongest predictor of delays
- Delayed orders receive significantly lower review scores
- Certain states have consistently higher delay rates