# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Configuração da página
st.set_page_config(
    page_title="Game Engagement Analysis",
    layout="centered",
    page_icon="🎮"
)

# Título do aplicativo
st.title("🎮 Análise de Engajamento em Jogos")


# Rodapé
st.markdown("---")
st.caption("Desenvolvido com Streamlit | Modelo treinado com PyCaret")
