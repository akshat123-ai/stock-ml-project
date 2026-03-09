import streamlit as st
import pandas as pd
import numpy as np
import pickle
import glob
import os
import plotly.express as px
import yfinance as yf
import shap
import requests

# =========================================================
# PATHS
# =========================================================

BASE_PATH = "D:/ML Lab/projects"

MODEL_PATH = os.path.join(BASE_PATH,"model.pkl")
SCALER_PATH = os.path.join(BASE_PATH,"scaler.pkl")
SELECTOR_PATH = os.path.join(BASE_PATH,"selector.pkl")
FEATURE_PATH = os.path.join(BASE_PATH,"selected_features.pkl")

DATASET_PATH = os.path.join(BASE_PATH,"dataset")

# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

model = pickle.load(open(MODEL_PATH,"rb"))
scaler = pickle.load(open(SCALER_PATH,"rb"))
selector = pickle.load(open(SELECTOR_PATH,"rb"))
selected_features = pickle.load(open(FEATURE_PATH,"rb"))

# =========================================================
# FULL FEATURE LIST
# =========================================================

FEATURES = [

"Return","MA5","MA10","MA_ratio",
"Volatility","Momentum",
"PriceRange","VolumeChange",
"Lag1","Lag2",

"EMA12","EMA26","MACD","RSI",

"BB_upper","BB_lower",

"OC_spread","HL_spread"

]

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Stock ML Dashboard", layout="wide")

st.title("📈 Stock Market ML Dashboard")

# =========================================================
# SIDEBAR
# =========================================================

page = st.sidebar.selectbox(
"Navigation",
[
"Dataset Overview",
"Model Comparison",
"Feature Importance",
"Strategy Simulation",
"Explainable AI (SHAP)",
"Live Stock Prediction",
"Prediction Demo"
]
)

# =========================================================
# FEATURE ENGINEERING FUNCTION
# =========================================================

def create_features(df):

    df["Return"] = df["Close"].pct_change()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()

    df["MA_ratio"] = df["MA5"] / df["MA10"]

    df["Volatility"] = df["Return"].rolling(5).std()

    df["Momentum"] = df["Close"].pct_change(5)

    df["PriceRange"] = (df["High"] - df["Low"]) / df["Close"]

    df["VolumeChange"] = df["Volume"].pct_change()

    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)

    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()

    df["MACD"] = df["EMA12"] - df["EMA26"]

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()

    df["BB_upper"] = rolling_mean + (2 * rolling_std)
    df["BB_lower"] = rolling_mean - (2 * rolling_std)

    df["OC_spread"] = (df["Open"] - df["Close"]) / df["Close"]
    df["HL_spread"] = (df["High"] - df["Low"]) / df["Close"]

    df.replace([np.inf,-np.inf],np.nan,inplace=True)

    df.dropna(inplace=True)

    return df

# =========================================================
# DATA LOADER
# =========================================================

@st.cache_data
def load_dataset():

    all_files = glob.glob(os.path.join(DATASET_PATH,"**","*.txt"),recursive=True)

    data_parts = []

    for file in all_files[:100]:

        try:

            df = pd.read_csv(file)

            df = df.sort_values("Date")

            df["Target"] = df["Close"].shift(-1)
            df["Target"] = (df["Target"] > df["Close"]).astype(int)

            df = create_features(df)

            data_parts.append(df)

        except:
            pass

    return pd.concat(data_parts,ignore_index=True)

# =========================================================
# DATASET OVERVIEW
# =========================================================

if page == "Dataset Overview":

    df = load_dataset()

    st.header("Dataset Overview")

    st.write("Dataset shape:",df.shape)

    st.dataframe(df.head())

    st.subheader("Class Distribution")

    st.bar_chart(df["Target"].value_counts().sort_index())

# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("Model Benchmark")

    data = {
        "Model":[
        "LightGBM","XGBoost","GradientBoosting","RandomForest",
        "LogisticRegression","DecisionTree","LinearSVM","MLP"
        ],

        "Accuracy":[0.53,0.53,0.52,0.52,0.52,0.51,0.51,0.50]
    }

    df = pd.DataFrame(data)

    st.bar_chart(df.set_index("Model"))

    st.dataframe(df)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.header("Feature Importance")

    if hasattr(model,"feature_importances_"):

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature":selected_features,
            "Importance":importance
        })

        imp_df = imp_df.sort_values("Importance",ascending=False)

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Model Feature Importance"
        )

        st.plotly_chart(fig)

# =========================================================
# STRATEGY SIMULATION
# =========================================================

elif page == "Strategy Simulation":

    st.header("Trading Strategy Simulation")

    df = load_dataset()

    X = df[FEATURES]

    X = selector.transform(X)

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)

    returns = df["Return"].values[:len(pred)]

    strategy_returns = pred * returns

    equity_curve = np.cumsum(strategy_returns)

    chart_df = pd.DataFrame({
        "Step":range(len(equity_curve)),
        "Equity":equity_curve
    })

    fig = px.line(chart_df,x="Step",y="Equity",title="Strategy Equity Curve")

    st.plotly_chart(fig)

    std = np.std(strategy_returns)

    sharpe = 0 if std==0 else np.mean(strategy_returns)/std

    drawdown = np.min(equity_curve-np.maximum.accumulate(equity_curve))

    st.metric("Sharpe Ratio",round(sharpe,3))
    st.metric("Max Drawdown",round(drawdown,3))

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

elif page == "Explainable AI (SHAP)":

    st.header("Explainable AI")

    from sklearn.inspection import permutation_importance

    df = load_dataset()

    X = df[FEATURES]

    # Apply selector
    X = selector.transform(X)

    # Scale features
    X_scaled = scaler.transform(X)

    # Use small sample for speed
    sample_X = X_scaled[:300]
    sample_y = df["Target"].values[:300]

    result = permutation_importance(
        model,
        sample_X,
        sample_y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance = result.importances_mean

    imp_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importance
    })

    imp_df = imp_df.sort_values("Importance", ascending=False)

    st.subheader("Feature Importance (Permutation)")

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig)

    st.dataframe(imp_df)

# =========================================================
# LIVE STOCK PREDICTION
# =========================================================

elif page == "Live Stock Prediction":

    st.header("Live Stock Prediction")

    ticker = st.text_input("Enter Stock Ticker","AAPL")

    if st.button("Fetch Data"):

        try:

            data = yf.download(ticker,period="6mo")

            if data.empty:
                st.error("Invalid ticker or no data available.")
                st.stop()

            data = create_features(data)

            if len(data) == 0:
                st.error("Not enough data after feature engineering.")
                st.stop()

            X = data[FEATURES]

            if X.shape[0] == 0:
                st.error("Feature matrix empty. Try another ticker.")
                st.stop()

            # Feature selection
            X = selector.transform(X)

            # Scaling
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)

            st.line_chart(data["Close"])

            signal = "UP 📈" if pred[-1] == 1 else "DOWN 📉"

            st.success(f"Latest Prediction: {signal}")

        except Exception as e:

            st.error("Prediction failed")
            st.code(str(e))

# =========================================================
# MANUAL PREDICTION
# =========================================================

elif page == "Prediction Demo":

    st.header("Manual Prediction")

    inputs = []

    for feature in selected_features:
        val = st.number_input(feature,value=0.0)
        inputs.append(val)

    if st.button("Predict"):

        payload = {"features": inputs}

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            result = response.json()

            if result["prediction"] == 1:
                st.success("📈 Stock likely to go UP")
            else:
                st.error("📉 Stock likely to go DOWN")

        except:
            st.error("⚠️ API server not running")
