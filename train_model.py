import pandas as pd
import numpy as np
import glob
import os
import zipfile
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================================================
# PATHS
# =========================================================

BASE_PATH = "D:/ML Lab/projects"

zip_path = os.path.join(BASE_PATH, "archive.zip")
extract_path = os.path.join(BASE_PATH, "dataset")


# =========================================================
# EXTRACT DATASET
# =========================================================

if not os.path.exists(extract_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Dataset extracted")


# =========================================================
# FIND FILES
# =========================================================

all_files = glob.glob(os.path.join(extract_path, "**", "*.txt"), recursive=True)

print("Total files found:", len(all_files))


# =========================================================
# LOAD DATA
# =========================================================

data_parts = []

MAX_FILES = 2000
ROWS_PER_STOCK = 600

for file in all_files[:MAX_FILES]:

    try:

        df = pd.read_csv(file)

        symbol = os.path.basename(file).replace(".us.txt", "")
        df["Symbol"] = symbol

        df = df.sort_values("Date")

        # TARGET
        df["Target"] = df["Close"].shift(-1)
        df["Target"] = (df["Target"] > df["Close"]).astype(int)

        # =====================================================
        # FEATURE ENGINEERING
        # =====================================================

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

        # ADVANCED FEATURES

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

        # CLEAN DATA

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) > ROWS_PER_STOCK:
            df = df.tail(ROWS_PER_STOCK)

        data_parts.append(df)

    except:
        pass


# =========================================================
# MERGE DATA
# =========================================================

full_df = pd.concat(data_parts, ignore_index=True)

print("Dataset shape:", full_df.shape)


# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

for col in full_df.select_dtypes(include=["float64"]).columns:
    full_df[col] = pd.to_numeric(full_df[col], downcast="float")

for col in full_df.select_dtypes(include=["int64"]).columns:
    full_df[col] = pd.to_numeric(full_df[col], downcast="integer")


# =========================================================
# FEATURES
# =========================================================

features = [

"Return","MA5","MA10","MA_ratio",
"Volatility","Momentum",
"PriceRange","VolumeChange",
"Lag1","Lag2",

"EMA12","EMA26","MACD","RSI",

"BB_upper","BB_lower",

"OC_spread","HL_spread"

]


X = full_df[features]
y = full_df["Target"]


# =========================================================
# CLEAN FEATURES
# =========================================================

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

y = y.loc[X.index]

X = X.clip(-10, 10)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# =========================================================
# RAM SAFE TRAINING SAMPLE
# =========================================================

TRAIN_SAMPLE = 500000

if len(X_train) > TRAIN_SAMPLE:

    sample_index = X_train.sample(TRAIN_SAMPLE, random_state=42).index

    X_train = X_train.loc[sample_index]
    y_train = y_train.loc[sample_index]


# =========================================================
# FEATURE SELECTION
# =========================================================

selector = SelectKBest(score_func=f_classif, k=15)

X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

selected_features = np.array(features)[selector.get_support()]

print("Selected Features:", selected_features)


# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# MODELS
# =========================================================

models = {

"LogisticRegression": LogisticRegression(max_iter=400, class_weight="balanced"),

"DecisionTree": DecisionTreeClassifier(max_depth=8),

"RandomForest": RandomForestClassifier(
    n_estimators=120,
    n_jobs=-1,
    class_weight="balanced"
),

"GradientBoosting": GradientBoostingClassifier(),

"LinearSVM": LinearSVC(),

"MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200),

"XGBoost": XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    n_jobs=-1
),

"LightGBM": LGBMClassifier(
    n_estimators=200,
    n_jobs=-1
)

}


scaled_models = [
"LogisticRegression",
"LinearSVM",
"MLP"
]


# =========================================================
# MODEL BENCHMARK
# =========================================================

results = []

best_model = None
best_score = 0
best_name = None

for name, model in models.items():

    print("\nTraining:", name)

    if name in scaled_models:

        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("Accuracy:", acc)
    print("F1:", f1)

    results.append((name, acc, f1))

    if acc > best_score:

        best_score = acc
        best_model = model
        best_name = name


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results, columns=["Model","Accuracy","F1"])

print("\nMODEL COMPARISON")
print(results_df.sort_values("Accuracy", ascending=False))

print("\nBest Model:", best_name)
print("Best Accuracy:", best_score)


# =========================================================
# WALK FORWARD VALIDATION
# =========================================================

tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []

sample = X_train_scaled[:30000]

for train_index, test_index in tscv.split(sample):

    Xtr = sample[train_index]
    Xte = sample[test_index]

    ytr = y_train.iloc[train_index]
    yte = y_train.iloc[test_index]

    best_model.fit(Xtr, ytr)

    pred = best_model.predict(Xte)

    cv_scores.append(accuracy_score(yte, pred))

print("\nWalk Forward Accuracy:", np.mean(cv_scores))


# =========================================================
# TRADING STRATEGY SIMULATION
# =========================================================

pred_test = best_model.predict(X_test_scaled)

returns = full_df["Return"].iloc[split:].values[:len(pred_test)]

strategy_returns = pred_test * returns

equity_curve = np.cumsum(strategy_returns)

print("\nStrategy Final Return:", equity_curve[-1])

std = np.std(strategy_returns)

if std == 0:
    sharpe = 0
else:
    sharpe = np.mean(strategy_returns) / std

max_drawdown = np.min(equity_curve - np.maximum.accumulate(equity_curve))

print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_drawdown)


# =========================================================
# SAVE ARTIFACTS
# =========================================================

pickle.dump(best_model, open(os.path.join(BASE_PATH,"model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(BASE_PATH,"scaler.pkl"), "wb"))
pickle.dump(selector, open(os.path.join(BASE_PATH,"selector.pkl"), "wb"))
pickle.dump(selected_features, open(os.path.join(BASE_PATH,"selected_features.pkl"), "wb"))

print("\nArtifacts saved successfully")
print("Best model:", best_name)