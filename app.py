
# ============================================================
# STORE CUSTOMER SEGMENTATION DASHBOARD
# Backend: Flask
# Algorithm: K-Means Clustering
# ============================================================

import os
import json
import pickle
import warnings
import pandas as pd

from flask import Flask, abort, render_template, request, send_file, flash

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "store_customers.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "customer_segmentation_results.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "customer_segmentation_model.pkl"
)

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

df = df.drop_duplicates().copy()

missing_columns = [
    col for col in FEATURES if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[FEATURES] = df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)

# Remove rows where all three features are missing
df = df.dropna(subset=FEATURES, how="all").copy()

# Fill remaining missing values
df[FEATURES] = df[FEATURES].fillna(
    df[FEATURES].median()
)

if len(df) < 11:
    raise ValueError(
        "Dataset must contain at least 11 valid customers."
    )

# ------------------------------------------------------------
# 3. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X = df[FEATURES]

X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 4. ELBOW METHOD
# ------------------------------------------------------------

inertia_values = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(
        round(float(model.inertia_), 2)
    )

# ------------------------------------------------------------
# 5. K-MEANS CLUSTERING
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "features": FEATURES
        },
        model_file
    )

# ------------------------------------------------------------
# 6. SILHOUETTE SCORE
# ------------------------------------------------------------

score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

# ------------------------------------------------------------
# 7. CLUSTER COUNT
# ------------------------------------------------------------

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 8. CLUSTER ANALYSIS
# ------------------------------------------------------------

cluster_analysis = (
    df.groupby("Cluster")[FEATURES]
    .mean()
    .round(2)
)

# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 10. CHART DATA
# ------------------------------------------------------------

scatter_data = []

for _, row in df.iterrows():

    scatter_data.append({
        "income": float(row["Annual Income (k$)"]),
        "spending": float(row["Spending Score (1-100)"]),
        "age": float(row["Age"]),
        "cluster": int(row["Cluster"])
    })

cluster_labels = [
    f"Cluster {int(c)}"
    for c in cluster_counts.index
]

cluster_values = [
    int(v)
    for v in cluster_counts.values
]

analysis_data = []

for cluster, row in cluster_analysis.iterrows():

    analysis_data.append({
        "cluster": int(cluster),
        "age": float(row["Age"]),
        "income": float(row["Annual Income (k$)"]),
        "spending": float(row["Spending Score (1-100)"])
    })

# ------------------------------------------------------------
# 11. DASHBOARD
# ------------------------------------------------------------

@app.route("/", defaults={"page": "dashboard"}, methods=["GET", "POST"])
@app.route("/<page>", methods=["GET", "POST"])
def dashboard(page):

    if page not in {"dashboard", "segmentation", "analysis", "prediction"}:
        abort(404)

    prediction = None

    if request.method == "POST":

        try:
            age = float(request.form["age"])
            income = float(request.form["income"])
            spending = float(request.form["spending"])

            if not all(map(lambda x: pd.notna(x) and x >= 0,
                           [age, income, spending])):
                raise ValueError("Enter valid non-negative values.")

            customer = pd.DataFrame(
                [[age, income, spending]],
                columns=FEATURES
            )

            customer_scaled = scaler.transform(customer)

            prediction = int(
                kmeans.predict(customer_scaled)[0]
            )

        except (ValueError, TypeError):
            flash("Please enter valid numeric customer details.")

    return render_template(
        "index.html",

        page=page,

        total_customers=len(df),

        total_clusters=5,

        silhouette=round(float(score), 4),

        avg_income=round(
            float(df["Annual Income (k$)"].mean()), 2
        ),

        avg_spending_score=round(
            float(df["Spending Score (1-100)"].mean()), 2
        ),

        avg_age=round(float(df["Age"].mean()), 2),

        median_income=round(
            float(df["Annual Income (k$)"].median()), 2
        ),

        max_spending_score=int(df["Spending Score (1-100)"].max()),

        largest_cluster_size=int(cluster_counts.max()),

        cluster_labels=json.dumps(cluster_labels),

        cluster_values=json.dumps(cluster_values),

        scatter_data=json.dumps(scatter_data),

        elbow_labels=json.dumps(list(range(2, 11))),

        elbow_values=json.dumps(inertia_values),

        analysis_data=analysis_data,

        prediction=prediction
    )

# ------------------------------------------------------------
# 12. DOWNLOAD CLUSTERED DATASET
# ------------------------------------------------------------

@app.route("/download")
def download():

    return send_file(
        OUTPUT_PATH,
        as_attachment=True,
        download_name="customer_segmentation_results.csv"
    )

# ------------------------------------------------------------
# 13. RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
