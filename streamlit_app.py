# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import load_model

# Configuração da página
st.set_page_config(
    page_title="Game Engagement Analysis",
    layout="centered",
    page_icon="🎮"
)

# Título do aplicativo
st.title("🎮 Análise de Engajamento em Jogos")
st.markdown("""
Análise preditiva baseada no modelo de machine learning treinado.
""")

# Carregar modelo (ajuste o nome se necessário)
@st.cache_resource
def load_ml_model():
    try:
        return load_model('game_eng')  # Assumindo que o modelo foi salvo com este nome
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None

model = load_ml_model()

# Seção de análise
st.header("📊 Resultados do Modelo")

if model:
    # Mostrar tipo do modelo
    st.subheader("Informações do Modelo")
    st.write(f"**Algoritmo:** {type(model).__name__}")
    
    # Exemplo de features importantes (ajuste conforme seu modelo)
    st.subheader("Fatores Importantes para Engajamento")
    st.markdown("""
    - Tempo de jogo (PlayTimeHours)
    - Nível do jogador (PlayerLevel)
    - Dificuldade do jogo (GameDifficulty)
    - Conquistas desbloqueadas (AchievementsUnlocked)
    """)
    
    # Gráfico explicativo
    st.subheader("Relação entre Variáveis")
    fig, ax = plt.subplots()
    sample_data = pd.DataFrame({
        'Variável': ['Tempo de Jogo', 'Nível', 'Dificuldade', 'Conquistas'],
        'Importância': [0.45, 0.3, 0.15, 0.1]  # Valores exemplos - substitua pelos reais
    })
    sns.barplot(data=sample_data, x='Importância', y='Variável', ax=ax)
    st.pyplot(fig)
else:
    st.warning("Modelo não encontrado. Verifique se o arquivo 'game_eng.pkl' existe.")

# Rodapé
st.markdown("---")
st.caption("""
Desenvolvido por [Seu Nome] | [Repositório GitHub](https://github.com/thaisar13)
""")
