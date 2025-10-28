# data_to_insights_app.py
# Streamlit app: Data-to-Insights Assistant (Enhanced with Auto Chart Intelligence)
# Requires: streamlit, pandas, matplotlib, seaborn, openai>=1.0.0

import os
import io
import tempfile
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ------------------------- Helpers -------------------------

def read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=';')


def get_basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile = {
        'n_rows': df.shape[0],
        'n_columns': df.shape[1],
        'columns': []
    }
    for col in df.columns:
        col_data = {
            'name': col,
            'dtype': str(df[col].dtype),
            'nulls': int(df[col].isna().sum()),
            'unique': int(df[col].nunique(dropna=True))
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data.update({
                'mean': float(df[col].mean(skipna=True)),
                'std': float(df[col].std(skipna=True)),
                'min': float(df[col].min(skipna=True)),
                'max': float(df[col].max(skipna=True)),
            })
        profile['columns'].append(col_data)
    return profile


def top_rows_sample(df: pd.DataFrame, n=5) -> str:
    return df.head(n).to_csv(index=False)


# ------------------------- Visualization -------------------------

def plot_numeric_histograms(df: pd.DataFrame, max_cols=6) -> List[str]:
    paths = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:max_cols]:
        fig, ax = plt.subplots()
        df[col].dropna().hist(bins=20, ax=ax)
        ax.set_title(f'Histogram: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('count')
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.tight_layout()
        fig.savefig(tmp.name)
        plt.close(fig)
        paths.append(tmp.name)
    return paths


def plot_categorical_bars(df: pd.DataFrame, max_cols=4, top_k=10) -> List[str]:
    paths = []
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols[:max_cols]:
        fig, ax = plt.subplots(figsize=(6, 3))
        counts = df[col].value_counts(dropna=True).nlargest(top_k)
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Top categories: {col}')
        ax.set_ylabel('count')
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.tight_layout()
        fig.savefig(tmp.name)
        plt.close(fig)
        paths.append(tmp.name)
    return paths


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax)
    ax.set_title('Correlation matrix')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.tight_layout()
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


# ------------------------- Auto Chart Intelligence -------------------------

def auto_generate_charts(df: pd.DataFrame) -> List[str]:
    charts = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Line chart (time-like column)
    time_like = [c for c in df.columns if 'month' in c.lower() or 'date' in c.lower()]
    if time_like and num_cols:
        for col in num_cols[:2]:
            for t in time_like[:1]:
                fig, ax = plt.subplots()
                df.groupby(t)[col].mean().plot(ax=ax, marker='o')
                ax.set_title(f"{col} trend over {t}")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                fig.tight_layout()
                fig.savefig(tmp.name)
                plt.close(fig)
                charts.append(tmp.name)

    # Bar chart (categorical vs numeric)
    for c in cat_cols[:2]:
        for n in num_cols[:2]:
            fig, ax = plt.subplots()
            df.groupby(c)[n].mean().nlargest(10).plot(kind='bar', ax=ax)
            ax.set_title(f"Average {n} by {c}")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            fig.tight_layout()
            fig.savefig(tmp.name)
            plt.close(fig)
            charts.append(tmp.name)

    # Scatter (first two numeric columns)
    if len(num_cols) >= 2:
        fig, ax = plt.subplots()
        ax.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.6)
        ax.set_xlabel(num_cols[0])
        ax.set_ylabel(num_cols[1])
        ax.set_title(f"{num_cols[0]} vs {num_cols[1]}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.tight_layout()
        fig.savefig(tmp.name)
        plt.close(fig)
        charts.append(tmp.name)

    return charts


# ------------------------- OpenAI Interaction -------------------------

def get_openai_client(key_from_sidebar: str) -> OpenAI:
    key = key_from_sidebar or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError("OpenAI API key not found. Set it in sidebar or environment variable.")
    return OpenAI(api_key=key)


def build_insight_prompt(profile, sample_csv, user_instructions=''):
    prompt = (
        "You are an expert data analyst. Summarize key insights from the dataset.\n"
        "Include: (1) Observations, (2) Possible actions, (3) Data quality notes.\n\n"
        f"Profile: {profile}\n\nSample:\n{sample_csv}\n\nUser question: {user_instructions}"
    )
    return prompt[:15000]


def ask_model_for_insights(client: OpenAI, prompt_text: str, model='gpt-4', temperature=0.2):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data scientist assistant."},
            {"role": "user", "content": prompt_text},
        ],
        temperature=temperature,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


# ------------------------- Streamlit UI -------------------------

def main():
    st.set_page_config(page_title='Data-to-Insights Assistant', layout='wide')
    st.title('📊 Data-to-Insights Assistant')
    st.write('Upload a CSV and get automatic data summaries, visualizations, and AI insights.')

    with st.sidebar:
        st.header('Settings & API')
        openai_key = st.text_input('OpenAI API Key', type='password')
        model = st.selectbox('Model', ['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'], index=0)
        temperature = st.slider('Response temperature', 0.0, 1.0, 0.2)
        show_prompt = st.checkbox('Show generated prompt', value=False)

    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
    if not uploaded_file:
        st.info('Upload a CSV file to begin (example: sales.csv, cricket_stats.csv).')
        st.stop()

    df = read_csv(uploaded_file)
    st.subheader('Preview')
    st.dataframe(df.head(20))

    profile = get_basic_profile(df)
    st.metric('Rows', profile['n_rows'])
    st.metric('Columns', profile['n_columns'])

    # Auto Visualization
    st.markdown('---')
    st.header('Auto Visualizations')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Numeric histograms')
        for p in plot_numeric_histograms(df):
            st.image(p)
    with col2:
        st.subheader('Categorical top-bars')
        for p in plot_categorical_bars(df):
            st.image(p)
    corr_path = plot_correlation_heatmap(df)
    if corr_path:
        st.subheader('Correlation heatmap')
        st.image(corr_path)

    # Auto Chart Intelligence (button-triggered)
    st.markdown('---')
    st.header('Auto Chart Intelligence')
    if st.button('Generate Auto Charts'):
        charts = auto_generate_charts(df)
        if charts:
            for c in charts:
                st.image(c)
        else:
            st.info('No meaningful charts could be auto-generated for this dataset.')

    # LLM Insights
    st.markdown('---')
    st.header('AI-Powered Insights')
    user_q = st.text_area('Optional: Ask a specific question about the data')
    if st.button('Generate Insights'):
        try:
            client = get_openai_client(openai_key)
        except Exception as e:
            st.error(str(e))
            st.stop()

        prompt = build_insight_prompt(profile, top_rows_sample(df, 6), user_q)
        if show_prompt:
            st.code(prompt)

        with st.spinner('Analyzing with LLM...'):
            try:
                result = ask_model_for_insights(client, prompt, model, temperature)
                st.markdown(result)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")


if __name__ == '__main__':
    main()
