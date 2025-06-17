# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import load_model
import os

# Configuração da página
st.set_page_config(
    page_title="Game Engagement Analysis",
    layout="wide",
    page_icon="🎮"
)

# Título do aplicativo
st.title("🎮 Game Engagement Analysis")
st.markdown("""
Análise preditiva de engajamento em jogos usando machine learning.
""")

# Funções para carregar dados e modelo com cache
@st.cache_data
def load_data():
    """Carrega os dados do repositório"""
    try:
        # Ajuste o caminho conforme sua estrutura no GitHub
        data_path = os.path.join('data', 'online_gaming_behavior_dataset.csv')
        dados = pd.read_csv(data_path)
        return dados[dados['EngagementLevel'].isin(['Low', 'High'])]
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_resource
def load_ml_model():
    """Carrega o modelo treinado"""
    try:
        model_path = os.path.join('models', 'melhor_modelo_dificuldade_jogo')
        return load_model(model_path)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Carregar dados e modelo
dados = load_data()
model = load_ml_model()

# Seção de Análise Exploratória
if dados is not None:
    with st.expander("🔍 Análise Exploratória dos Dados", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dados Brutos")
            st.dataframe(dados.head(), use_container_width=True)
            
            st.subheader("Estatísticas")
            st.dataframe(dados.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Distribuição de Engajamento")
            fig, ax = plt.subplots()
            sns.countplot(data=dados, x='EngagementLevel', ax=ax)
            st.pyplot(fig)
            
            st.subheader("Idade vs. Tempo de Jogo")
            fig, ax = plt.subplots()
            sns.scatterplot(data=dados, x='Age', y='PlayTimeHours', hue='EngagementLevel', ax=ax)
            st.pyplot(fig)

# Seção do Modelo
if model is not None:
    with st.expander("🤖 Resultados do Modelo", expanded=True):
        st.subheader("Informações do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Modelo Treinado:**
            - Algoritmo: CatBoost (provavelmente)
            - Métrica otimizada: F1-Score
            - Variável alvo: EngagementLevel
            """)
            
        with col2:
            st.markdown("""
            **Variáveis Importantes:**
            1. PlayTimeHours
            2. PlayerLevel
            3. GameDifficulty_Hard
            4. AchievementsUnlocked
            5. Age
            """)

# Configuração para GitHub
st.markdown("---")
st.markdown("""
**Configuração do Repositório:**
