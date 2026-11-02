# Customer Segmentation

A Flask dashboard that segments customers with a five-cluster K-Means model.

**Aspiring Data Analyst & Data Scientist – Haritha K.**

## Run Locally

1. Install dependencies with `pip install -r requirements.txt`.
2. Place the input file `store_customers.csv` in the project root. It must include `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` columns.
3. Run `python app.py` and open the local address printed by Flask.

The customer CSV and generated results CSV are intentionally excluded from Git. On startup, the app writes `customer_segmentation_model.pkl`, containing the fitted scaler, K-Means model, and feature names.

Only load the pickle file if you trust its source. Set `FLASK_SECRET_KEY` to a stable, private value when deploying the app; otherwise, a random key is generated for each run.
