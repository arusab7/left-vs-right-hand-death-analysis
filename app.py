import os
import io
import zipfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import streamlit as st

st.set_page_config(page_title="Left vs Right Handed Analysis", layout="wide")
sns.set_style("whitegrid")

def load_csv():
    if os.path.exists("data_set.csv"):
        return pd.read_csv("data_set.csv")
    return None

# ---- UI ----
st.title("Death Age Analysis — Right vs Left Handers")

uploaded = st.file_uploader("Upload data_set.csv (optional)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_csv()

if df is None:
    st.error("No CSV found. Please upload data_set.csv")
    st.stop()

# ---- RENAME COLUMNS ----
df = df.rename(columns={
    "Male": "RightHanded",
    "Female": "LeftHanded"
})

# ---- PREVIEW ----
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---- STATISTICS ----
r_mean = df["RightHanded"].mean()
l_mean = df["LeftHanded"].mean()
diff = r_mean - l_mean
corr = df["RightHanded"].corr(df["LeftHanded"])
tstat, pval = stats.ttest_ind(df["RightHanded"], df["LeftHanded"], equal_var=False)

st.subheader("Statistics Summary")
st.write(df.describe())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Right-Handed Mean", f"{r_mean:.4f}")
c2.metric("Left-Handed Mean", f"{l_mean:.4f}")
c3.metric("Difference (R-L)", f"{diff:.4f}")
c4.metric("Correlation", f"{corr:.4f}")

st.write(f"T-test: t = {tstat:.4f}, p = {pval:.6f}")

# ---- PLOTS ----
def plot_line():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Age"], df["RightHanded"], marker="o", label="Right-Handed")
    ax.plot(df["Age"], df["LeftHanded"], marker="o", label="Left-Handed")
    ax.set_title("Right vs Left Handed Across Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Values")
    ax.legend()
    ax.grid(True)
    return fig

def plot_scatter():
    fig, ax = plt.subplots(figsize=(6,6))
    sns.regplot(x="RightHanded", y="LeftHanded", data=df, ax=ax)
    ax.set_title("Right vs Left Handed — Regression Plot")
    return fig

def plot_hist():
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    df["RightHanded"].hist(ax=axes[0])
    axes[0].set_title("Right-Handed Distribution")
    df["LeftHanded"].hist(ax=axes[1])
    axes[1].set_title("Left-Handed Distribution")
    return fig

def plot_heat():
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(df[["RightHanded", "LeftHanded"]].corr(), annot=True, cmap="coolwarm")
    return fig

st.subheader("Visualizations")
st.pyplot(plot_line())
st.pyplot(plot_scatter())
st.pyplot(plot_hist())
st.pyplot(plot_heat())

