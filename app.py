import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Fraud Detection Model Demo")
st.write("A streamlined demo of the fraud detection project, including EDA, model insights, and predictions.")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data\cleaned_fraud_data.csv")  # <-- replace with your file
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df



# Feature engineering for SHAP (must match training)
def add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["second"] = df["timestamp"].dt.second

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df

df = load_data()
df = add_time_features(df)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.pkl")  # <-- your saved model
    return model

model = load_model()






import shap

@st.cache_resource
def load_shap_explainer(_model, df):
    """
    Build and cache the SHAP explainer and preprocessor.

    Assumes your pipeline looks like:
    Pipeline(steps=[('preprocessor', ColumnTransformer(...)),
                    ('model', LogisticRegression(...))])
    """
    # Extract the preprocessing step and the inner LR model
    preprocessor = model.named_steps["preprocess"]   # <-- name from your pipeline
    lr_model = model.named_steps["classifier"]              # <-- name from your pipeline

    # Use a sample of the data to initialise SHAP background
    X = df.drop(columns=["is_fraud"])
    X_sample = X.sample(300, random_state=42)
    X_transformed = preprocessor.transform(X_sample)

    # Create SHAP explainer for a linear model
    explainer = shap.LinearExplainer(lr_model, X_transformed)

    return explainer, preprocessor

# Make them globally available in the app
explainer, preprocessor = load_shap_explainer(model, df)




# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Project Overview", "EDA Insights", "Model Performance", "Try a Prediction","Explainability" ]
)

# ---------------------------------------------------------
# PAGE 1 — PROJECT OVERVIEW
# ---------------------------------------------------------
if page == "Project Overview":
    st.header("📌 Project Overview")

    st.subheader("Problem We’re Solving")
    st.write("""
    Fraudulent transactions cause financial loss, operational overhead, and customer distrust.
    Fraud is rare, adaptive, and difficult to detect manually.
    Our goal is to build a machine‑learning model that detects fraud early and accurately.
    """)

    st.subheader("Core Decisions")
    st.write("""
    - Prioritised **recall** to minimise missed fraud.
    - Engineered behavioural, temporal, and risk‑based features.
    - Used Logistic Regression and Random Forest.
    - Tuned models using Random Search.
    - Evaluated using recall, precision, F1, and ROC‑AUC.
    """)

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head())

# ---------------------------------------------------------
# PAGE 2 — EDA INSIGHTS
# ---------------------------------------------------------
elif page == "EDA Insights":
    st.header("📊 EDA Insights")

    # 1. Class Distribution
    st.subheader("Class Distribution: Fraud vs Non-Fraud")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df["is_fraud"], palette="viridis", ax=ax)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

    # 2. Transaction Amount Distribution
    st.subheader("Transaction Amount Distribution (Log Scale)")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(
        data=df,
        x="amount_usd",
        hue="is_fraud",
        bins=50,
        kde=True,
        log_scale=True,
        palette="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    # 3. Boxplot
    st.subheader("Transaction Amounts: Fraud vs Non-Fraud")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df, x="is_fraud", y="amount_usd", palette="coolwarm", ax=ax)
    ax.set_yscale("log")
    st.pyplot(fig)

    # 4. Fraud Rate by Hour
    st.subheader("Fraud Rate by Hour of Day")
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")["is_fraud"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax)
    st.pyplot(fig)

    # 5. Fraud Rate by Day of Week
    st.subheader("Fraud Rate by Day of Week")
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    dow = df.groupby("day_of_week")["is_fraud"].mean()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=dow.index, y=dow.values, palette="magma", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------
# PAGE 3 — MODEL PERFORMANCE
# ---------------------------------------------------------
elif page == "Model Performance":
    st.header("🤖 Model Performance")

    st.write("Below are the evaluation results for Logistic Regression and Random Forest.")

    # Load precomputed results (replace with your actual values)
    lr_results = {
        "Accuracy": 0.9806,
        "Precision": 1.0,
        "Recall": 0.7789,
        "F1 Score": 0.8757,
        "ROC-AUC": 0.9560
    }

    rf_results = {
        "Accuracy": 0.9704,
        "Precision": 1.0,
        "Recall": 0.6631,
        "F1 Score": 0.7976,
        "ROC-AUC": 0.9181
    }

    st.subheader("Logistic Regression Metrics")
    st.json(lr_results)

    st.subheader("Random Forest Metrics")
    st.json(rf_results)

    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Logistic Regression")
        cm_lr = np.array([[2068, 0], [44, 155]])
        fig, ax = plt.subplots()
        sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Random Forest")
        cm_rf = np.array([[2068, 0], [67, 132]])
        fig, ax = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------
# PAGE 4 — PREDICTION DEMO (REAL MODEL)
# ---------------------------------------------------------
elif page == "Try a Prediction":
    st.header("🔮 Try a Fraud Prediction")

    st.write("Enter transaction details below to get a fraud prediction using the trained Logistic Regression model.")

    # Collect inputs
    amount_usd = st.number_input("Transaction Amount (USD)", min_value=0.0, value=100.0)
    amount_src = amount_usd
    fee = st.number_input("Fee", min_value=0.0, value=2.0)
    exchange_rate = st.number_input("Exchange Rate", min_value=0.0, value=1.0)

    hour = st.slider("Hour of Day", 0, 23, 12)
    minute = 0
    second = 0

    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
    ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.2)
    corridor_risk = st.slider("Corridor Risk", 0.0, 1.0, 0.1)

    # Required categorical inputs
    home_country = st.selectbox("Home Country", ["US", "CA", "UK"])
    source_currency = st.selectbox("Source Currency", ["USD", "CAD", "GBP"])
    dest_currency = st.selectbox("Destination Currency", ["USD", "CAD", "GBP"])
    channel = st.selectbox("Channel", ["web", "mobile", "atm"])
    ip_country = st.selectbox("IP Country", ["US", "CA", "UK"])
    kyc_tier = st.selectbox("KYC Tier", ["standard", "enhanced"])

    # NEW: Add IP address input
    ip_address = st.text_input("IP Address", value="0.0.0.0")

    # Other required fields
    new_device = st.checkbox("New Device?")
    location_mismatch = st.checkbox("Location Mismatch?")
    account_age_days = st.number_input("Account Age (days)", min_value=0, value=200)
    chargeback_history_count = st.number_input("Chargeback Count", min_value=0, value=0)
    risk_score_internal = st.slider("Internal Risk Score", 0.0, 1.0, 0.2)
    txn_velocity_1h = st.number_input("Txn Velocity (1h)", min_value=0, value=0)
    txn_velocity_24h = st.number_input("Txn Velocity (24h)", min_value=0, value=0)

    # Build timestamp features
    year = 2024
    month = 1
    day = 1
    day_of_week = 1
    day_of_year = 1
    week_of_year = 1
    is_weekend = 0

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Missingness flags
    ip_country_missing = 0
    kyc_tier_missing = 0
    device_trustScore_missing = 0

    # Build full input row
    input_df = pd.DataFrame([{
        "amount_usd": amount_usd,
        "amount_src": amount_src,
        "fee": fee,
        "exchange_rate_src_to_dest": exchange_rate,
        "hour": hour,
        "minute": minute,
        "second": second,
        "year": year,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "week_of_year": week_of_year,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "device_trust_score": device_trust_score,
        "ip_risk_score": ip_risk_score,
        "corridor_risk": corridor_risk,
        "home_country": home_country,
        "source_currency": source_currency,
        "dest_currency": dest_currency,
        "channel": channel,
        "ip_country": ip_country,
        "kyc_tier": kyc_tier,
        "new_device": new_device,
        "location_mismatch": location_mismatch,
        "account_age_days": account_age_days,
        "chargeback_history_count": chargeback_history_count,
        "risk_score_internal": risk_score_internal,
        "txn_velocity_1h": txn_velocity_1h,
        "txn_velocity_24h": txn_velocity_24h,
        "ip_country_missing": ip_country_missing,
        "kyc_tier_missing": kyc_tier_missing,
        "device_trustScore_missing": device_trustScore_missing,
        "ip_address": ip_address
    }])

    if st.button("Predict Fraud"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.write(f"Fraud Probability: **{probability:.4f}**")

        if prediction == 1:
            st.error("⚠️ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction appears legitimate.")


# ==========================================================
# Explainability 
#==========================================================


elif page == "Explainability":
    st.header("🧠 Model Explainability (SHAP)")
    st.write("""
    This page explains **why the model makes certain predictions** using SHAP.
    SHAP shows how each feature pushes a prediction toward **fraud** or **legitimate**.
    """)

    # ---------------------------------------------------------
    # 1. PREPARE DATA FOR SHAP
    # ---------------------------------------------------------
    # SHAP must operate on the same features the model was trained on.
    # We drop the target column and keep only input features.
    X = df.drop(columns=["is_fraud"])

    # Transform the raw features using the pipeline's preprocessor
    # This ensures SHAP sees the same scaled/encoded features the model sees.
    X_transformed = preprocessor.transform(X)

    # Compute SHAP values for the entire dataset
    shap_values = explainer.shap_values(X_transformed)

    # ---------------------------------------------------------
    # 2. GLOBAL SHAP SUMMARY PLOT
    # ---------------------------------------------------------
    st.subheader("🌍 Global Feature Importance")

    st.write("""
    This plot shows which features have the **strongest overall impact** on fraud predictions.
    - **Red** = pushes prediction toward fraud  
    - **Blue** = pushes prediction toward legitimate  
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=preprocessor.get_feature_names_out(),
        show=False
    )
    st.pyplot(fig)

    # ---------------------------------------------------------
    # 3. LOCAL EXPLANATION FOR A SINGLE TRANSACTION
    # ---------------------------------------------------------
    st.subheader("🔍 Explain a Single Prediction")

    st.write("Select a transaction index to understand why the model predicted fraud or legitimate.")

    # User selects a row from the dataset
    index = st.number_input(
        "Select a transaction index",
        min_value=0,
        max_value=len(df) - 1,
        value=10
    )

    # Extract the selected row
    row = X.iloc[[index]]

    # Transform it using the preprocessor
    row_transformed = preprocessor.transform(row)

    # Compute SHAP values for this row
    row_shap = explainer.shap_values(row_transformed)[0]

    # ---------------------------------------------------------
    # 3A. SHAP FORCE PLOT
    # ---------------------------------------------------------
    st.write("### SHAP Force Plot")
    st.write("Shows how each feature pushes the prediction toward fraud or legitimate.")

    shap_fig = shap.force_plot(
        explainer.expected_value,
        row_shap,
        row_transformed,
        feature_names=preprocessor.get_feature_names_out(),
        matplotlib=True
    )
    st.pyplot(shap_fig)

    # ---------------------------------------------------------
    # 3B. SHAP WATERFALL PLOT
    # ---------------------------------------------------------
    st.write("### SHAP Waterfall Plot")
    st.write("Shows the step-by-step contribution of each feature to the final prediction.")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        row_shap,
        feature_names=preprocessor.get_feature_names_out(),
        max_display=15,
        show=False
    )
    st.pyplot(fig2)

    # ---------------------------------------------------------
    # 4. NATURAL LANGUAGE EXPLANATION
    # ---------------------------------------------------------
    st.subheader("🗣 Why Was This Flagged?")

    # Build a dataframe of feature contributions
    shap_df = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
        "shap_value": row_shap
    }).sort_values("shap_value", ascending=False)

    # Top 3 features increasing fraud risk
    top_positive = shap_df.head(3)

    # Top 3 features decreasing fraud risk
    top_negative = shap_df.tail(3)

    st.write("### 🚩 Factors Increasing Fraud Risk")
    for _, r in top_positive.iterrows():
        st.write(f"- **{r['feature']}** increased fraud likelihood")

    st.write("### 🛡 Factors Reducing Fraud Risk")
    for _, r in top_negative.iterrows():
        st.write(f"- **{r['feature']}** reduced fraud likelihood")

    st.write("""
    These explanations help compliance teams, analysts, and stakeholders understand  
    **why the model made this decision**, improving trust and transparency.
    """)
