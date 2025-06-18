# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuração da página (APENAS UMA VEZ no início do arquivo)
st.set_page_config(
    page_title="Game Engagement Analysis",
    layout="centered",
    page_icon="🎮"
)

# Título do aplicativo
st.title("🎮 Análise de Engajamento em Jogos")
st.markdown("Análise preditiva baseada no modelo de machine learning treinado.")

# --- Carregamento do Modelo ---
# Caminho corrigido para o Streamlit Sharing
model_path = os.path.join('model.pkl')

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Modelo carregado com sucesso!")
        
        # Seção de análise (só mostra se o modelo carregar)
        st.header("📊 Resultados do Modelo")
        
        # Informações do Modelo
        st.subheader("Informações do Modelo")
        st.write(f"**Algoritmo:** {type(model).__name__}")
        
        # Features importantes (ajuste com os valores reais do seu modelo)
        st.subheader("Fatores Importantes para Engajamento")
        st.markdown("""
        - Tempo de jogo (PlayTimeHours)
        - Nível do jogador (PlayerLevel)
        - Dificuldade do jogo (GameDifficulty)
        - Conquistas desbloqueadas (AchievementsUnlocked)
        """)
        
        # Gráfico (valores exemplos - substitua pelos reais)
        st.subheader("Relação entre Variáveis")
        fig, ax = plt.subplots()
        sample_data = pd.DataFrame({
            'Variável': ['Tempo de Jogo', 'Nível', 'Dificuldade', 'Conquistas'],
            'Importância': [0.45, 0.3, 0.15, 0.1]  
        })
        sns.barplot(data=sample_data, x='Importância', y='Variável', ax=ax)
        st.pyplot(fig)
        
    else:
        st.error(f"Erro: Modelo não encontrado em {model_path}")
        st.write("Arquivos disponíveis:", os.listdir('models'))
        
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {str(e)}")
    st.write("Detalhes técnicos:", e)

# Rodapé
st.markdown("---")
st.caption("Desenvolvido por [Seu Nome] | [Repositório GitHub](https://github.com/thaisar13)")
