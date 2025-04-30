import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz
import os
import base64
import tempfile

st.set_page_config(page_title="📊 Automated EDA App", layout="wide")

st.title("📁 Upload Dataset for Automated Analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


# 🔧 Custom function to embed YData Profiling report in Streamlit
def st_profile_report(pr):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        pr.to_file(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        b64 = base64.b64encode(html.encode("utf-8")).decode()
        st.markdown(f'<iframe src="data:text/html;base64,{b64}" width="1000" height="600"></iframe>', unsafe_allow_html=True)

# 🔥 Heatmap
def plot_heatmap(df):
    st.subheader("🔍 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns to display a heatmap.")

# 📊 Plots Section
def plot_visualizations(df):
    st.subheader("📊 Univariate Analysis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        st.write(f"### 📈 Distribution Plot for: {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    for col in categorical_cols:
        st.write(f"### 📊 Count Plot for: {col}")
        fig, ax = plt.subplots()
        df[col] = df[col].astype(str)
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
        st.pyplot(fig)

    st.subheader("🧰 Box Plots (Outlier Detection)")
    for col in numeric_cols:
        st.write(f"### 📦 Box Plot for: {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

# 📋 YData-Profiling Report
def generate_profile_report(df):
    st.subheader("📋 YData Profiling Report")
    profile = ProfileReport(df, explorative=True)
    st_profile_report(profile)

# 🍭 Sweetviz Report
def generate_sweetviz_report(df):
    st.subheader("🍭 Sweetviz Report")
    report = sweetviz.analyze(df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        report.show_html(filepath=tmp.name, open_browser=False)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()

    b64 = base64.b64encode(html.encode("utf-8")).decode()
    st.markdown(f'<iframe src="data:text/html;base64,{b64}" width="1000" height="600"></iframe>', unsafe_allow_html=True)

# 🟢 Main Execution
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.write(df.head())

    st.subheader("📐 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("🧠 Column Info")
    st.write(df.dtypes)

    plot_heatmap(df)
    plot_visualizations(df)
    generate_profile_report(df)
    generate_sweetviz_report(df)
else:
    st.warning("👆 Upload a CSV file to get started!")
