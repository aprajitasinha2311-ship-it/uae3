
import plotly.express as px
import streamlit as st
import pandas as pd

def show_metrics(metrics):
    st.write(metrics)

def show_feature_importance(fi):
    df = pd.DataFrame({"importance":fi})
    st.bar_chart(df)

def show_clusters(df, X):
    from sklearn.cluster import KMeans
    k = KMeans(n_clusters=3)
    df["cluster"] = k.fit_predict(X)
    fig = px.scatter(df, x=X.columns[0], y=X.columns[1], color="cluster")
    st.plotly_chart(fig)
