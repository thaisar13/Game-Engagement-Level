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

@st.cache_data
def load_and_preprocess():
    """Carrega e pré-processa os dados (igual ao seu original)"""
    try:
        # Download dos dados
        path = kagglehub.dataset_download("rabieelkharoua/predict-online-gaming-behavior-dataset")
        csv_path = os.path.join(path, "online_gaming_behavior_dataset.csv")
        dados = pd.read_csv(csv_path)
        
        # Pré-processamento
        dados = dados[dados['EngagementLevel'].isin(['Low', 'High'])]
        dados.drop(['PlayerID', 'AvgSessionDurationMinutes','Gender', 'Location'], axis=1, inplace=True)
        dados['EngagementLevel'] = dados['EngagementLevel'].map({'Low': 0, 'High': 1})
        
        # Dummyficação
        variaveis_cat = ['GameGenre', 'GameDifficulty', 'InGamePurchases']
        dados = pd.get_dummies(dados, columns=variaveis_cat, drop_first=True)
        
        # Escalonamento
        scaler = StandardScaler()
        numericas = ['Age', 'SessionsPerWeek', 'PlayTimeHours', 'AchievementsUnlocked', 'PlayerLevel']
        dados[numericas] = scaler.fit_transform(dados[numericas])
        
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    try:
        # Caminho para o modelo (ajuste conforme necessário)
        model_path = 'models/melhor_modelo_dificuldade_jogo.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.error("Modelo não encontrado!")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None

# Interface principal
dados = load_and_preprocess()
model = load_model()

if dados is not None and model is not None:
    st.success("✅ Dados e modelo carregados com sucesso!")
    
    # Seção de análise
    st.header("📊 Resultados do Modelo")
    st.subheader("Fatores Importantes")
    
    # Gráfico de importância (valores exemplos)
    fig, ax = plt.subplots()
    features = pd.DataFrame({
        'Feature': ['PlayTime', 'PlayerLevel', 'Difficulty', 'Achievements'],
        'Importance': [0.45, 0.30, 0.15, 0.10]
    })
    sns.barplot(data=features, x='Importance', y='Feature', ax=ax)
    st.pyplot(fig)
    
    # Seção de previsão
    st.header("🔮 Fazer Previsão")
    
    with st.form("prediction_form"):
        age = st.number_input("Idade", min_value=10, max_value=80)
        sessions = st.number_input("Sessões por semana", min_value=1, max_value=50)
        playtime = st.number_input("Horas jogadas", min_value=1, max_value=100)
        level = st.number_input("Nível", min_value=1, max_value=100)
        
        submitted = st.form_submit_button("Prever")
        
        if submitted:
            # Criar input no mesmo formato dos dados de treino
            input_data = pd.DataFrame({
                'Age': [age],
                'SessionsPerWeek': [sessions],
                'PlayTimeHours': [playtime],
                'PlayerLevel': [level],
                'AchievementsUnlocked': [0],  # Valor padrão
                'GameDifficulty_Hard': [1],   # Valor padrão
                'GameGenre_Adventure': [0],
                'GameGenre_RPG': [0],
                'GameGenre_Strategy': [0],
                'InGamePurchases_Occasional': [0],
                'InGamePurchases_Frequent': [0]
            })
            
            # Fazer previsão
            prediction = model.predict(input_data)
            result = "Alto Engajamento" if prediction[0] == 1 else "Baixo Engajamento"
            st.success(f"Resultado: {result}")

# Rodapé
st.markdown("---")
st.caption("Desenvolvido com Streamlit | Modelo treinado com PyCaret")
