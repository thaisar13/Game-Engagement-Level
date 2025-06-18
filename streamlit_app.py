# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

#Sua aplicação Streamlit deve conter as seguintes seções:
#Explicação do Problema: Detalhar o problema de ML e apresentar o conjunto de dados.
#Processo e Justificativa: Descrever e justificar as escolhas no pré-processamento dos dados e na seleção do modelo.
#Implantação do Modelo: A aplicação deve permitir a interação do usuário para obter previsões com o modelo selecionado.

st.set_page_config(
    page_title="Game Engagement Analysis",
    layout="centered",
    page_icon="🎮"
)

# Título do aplicativo
st.title("🎮 Análise de Engajamento em Jogos")
st.markdown("Análise preditiva baseada no modelo de machine learning treinado.")


st.header("📊 Descrição do Problema")
st.markdown("""

O pricipal objetivo desse trabalho foi o de criar um modelo capaz de prever o engajamento de um jogo, 
com base nas caracteristicas dos jogadores. E para isso foi utilizada a base de dados *online_gaming_behavior_dataset*
disponivel no Kaggle[https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset].

""")

# adicionar uma breve descritiva dos dados
st.header("📊 Conjunto de Dados")
st.markdown("""
## """)

# colocar um barplot das contagens de cada categoria dos dados
st.subheader("Contagem de Categorias")
fig, ax = plt.subplots(figsize=(10, 6))
dados['EngagementLevel'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

colunas = ['Age', 'SessionsPerWeek', 'PlayTimeHours', 'AchievementsUnlocked', 'PlayerLevel', 'GenderGame']
st.subheader("Distribuição das Variáveis")
for i in colunas:
    fig, ax = plt.subplots(figsize=(10, 6))
    dados[i].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
fig, ax = plt.subplots(figsize=(10, 6))
dados['EngagementLevel'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

# colocar uma matriz de correlação das variaveis continuas
st.subheader("Matriz de Correlação")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(dados.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)



st.header("📊 Processo e Justificativa")  

st.markdown("""
## """)


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
