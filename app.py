import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

st.set_page_config("End-to-End Linear Regression", layout="wide")
st.title("End-to-End Linear Regression Platform")

st.header("Step 1 : Data Ingestion")

@st.cache_data
def load_data():
    data = load_diabetes(as_frame=True)
    df = data.frame
    df['target'] = data.target

    np.random.seed(42)

    for col in df.columns[:-1]:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    return df

df = load_data()

raw_path = os.path.join(RAW_DIR, "diabetes_raw.csv")
df.to_csv(raw_path, index=False)

st.success("Dataset Loaded")
st.dataframe(df.head())

# ================================
# EDA
# ================================

st.header("Step 2 : Exploratory Data Analysis")

st.write("Shape:", df.shape)

st.write("Missing Values")
st.write(df.isnull().sum())

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ================================
# DATA CLEANING
# ================================

st.header("Step 3 : Data Cleaning")

strategy = st.selectbox(
    "Missing Value Strategy",
    ["Mean", "Median", "Drop Rows"]
)

df_clean = df.copy()

if strategy == "Drop Rows":
    df_clean = df_clean.dropna()

elif strategy == "Mean":
    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].mean()
    )

elif strategy == "Median":
    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].median()
    )

st.subheader("Cleaned Dataset")
st.dataframe(df_clean.head())

st.subheader("Remaining Missing Values")
st.write(df_clean.isnull().sum())

if st.button("Save Cleaned Dataset"):
    clean_path = os.path.join(CLEAN_DIR, "cleaned_diabetes.csv")
    df_clean.to_csv(clean_path, index=False)

    st.success("Dataset Saved")
    st.info(f"Saved at: {clean_path}")

# ================================
# LOAD CLEANED DATA
# ================================

st.header("Step 4 : Load Cleaned Dataset")

clean_files = [f for f in os.listdir(CLEAN_DIR) if "diabetes" in f.lower()]

if not clean_files:
    st.error("No cleaned dataset found")
    st.stop()

selected = st.selectbox("Select Dataset", clean_files)

df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

st.dataframe(df_model.head())

# ================================
# TRAIN MODEL
# ================================

st.header("Step 5 : Train Linear Regression Model")

target = "target"

X = df_model.drop(columns=[target])
y = df_model[target]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ================================
# EVALUATION
# ================================

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.success(f"R² Score: {r2:.2f}")
st.success(f"MSE: {mse:.2f}")

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred, alpha=0.6)

ax.set_xlabel("Actual Target")
ax.set_ylabel("Predicted Target")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)