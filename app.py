
import streamlit as st
from utils import load_data, preprocess
from model import train_models, predict_user
from visuals import show_metrics, show_feature_importance, show_clusters

st.title("StepHome UAE - Decision Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = load_data(file)
    df_clean, X, y_conv, y_def = preprocess(df)
    model_conv, model_def, metrics, fi = train_models(X, y_conv, y_def)

    show_metrics(metrics)
    show_feature_importance(fi)
    show_clusters(df_clean, X)

    st.subheader("Predict New Customer")
    user = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}

    if st.button("Predict"):
        conv, risk = predict_user(user, model_conv, model_def)
        st.write("Conversion:", conv)
        st.write("Default Risk:", risk)
