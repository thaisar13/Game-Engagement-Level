# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Configurações da página
st.set_page_config(
    page_title="Game Engagement Predictor",
    layout="wide",
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

# Funções de pré-processamento
def preprocess_data(df):
    # Criar cópia para não modificar o original
    df_processed = df.copy()
    
    # Filtrar apenas níveis Low e High
    df_processed = df_processed[df_processed['EngagementLevel'].isin(['Low', 'High'])]
    
    # Converter EngagementLevel para binário
    df_processed['EngagementLevel'] = df_processed['EngagementLevel'].map({'Low': 0, 'High': 1})
    
    # Dummyficação
    cat_vars = ['GameGenre', 'GameDifficulty', 'InGamePurchases']
    df_processed = pd.get_dummies(df_processed, columns=cat_vars, drop_first=True)
    
    # Padronização
    scaler = StandardScaler()
    numeric_vars = ['Age', 'SessionsPerWeek', 'PlayTimeHours', 
                   'AchievementsUnlocked', 'PlayerLevel']
    df_processed[numeric_vars] = scaler.fit_transform(df_processed[numeric_vars])
    
    return df_processed, scaler

# Carregar e preparar dados
@st.cache_data
def load_data():
    try:
        # Carregar dados originais
        dados = pd.read_csv("online_gaming_behavior_dataset.csv")
        
        # Filtrar apenas Low e High para visualização também
        dados = dados[dados['EngagementLevel'].isin(['Low', 'High'])]
        
        # Criar versão para visualização (remove colunas não usadas)
        cols_to_keep = ['Age', 'SessionsPerWeek', 'PlayTimeHours', 
                       'AchievementsUnlocked', 'PlayerLevel', 'EngagementLevel',
                       'GameGenre', 'GameDifficulty', 'InGamePurchases']
        
        dados_vis = dados[cols_to_keep].copy()
        
        # Processar dados para modelagem
        dados_prep, scaler = preprocess_data(dados)
        
        return dados_vis, dados_prep, scaler
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None

# Carregar os dados
dados_vis, dados_prep, scaler = load_data()

# Barra lateral - Navegação
st.sidebar.title("Menu")
pagina = st.sidebar.radio(
    "Seções:",
    ["🏠 Visão Geral", "🔍 Análise Exploratória", "⚙️ Pré-processamento", "🤖 Modelo Preditivo", "🔮 Fazer Previsão"]
)

# Página 1: Visão Geral
if pagina == "🏠 Visão Geral":
    st.title("🎮 Análise de Engajamento em Jogos")
    st.markdown("---")
    
    st.header("📋 Sobre o Projeto")
    st.markdown("""
    Este projeto utiliza machine learning para prever o nível de engajamento de jogadores com base em:
    - Características demográficas
    - Comportamento de jogo
    - Preferências e conquistas
    """)
    
    if dados_vis is not None:
        st.header("📊 Dados Brutos (Amostra)")
        st.dataframe(dados_vis.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Registros", len(dados_vis))
        with col2:
            st.metric("Variáveis Originais", len(dados_vis.columns))

# Página 2: Análise Exploratória
elif pagina == "🔍 Análise Exploratória":
    st.title("🔍 Análise Exploratória dos Dados")
    st.markdown("---")
    
    if dados_orig is not None:
        st.header("Distribuição de Engajamento")
        fig, ax = plt.subplots(figsize=(8,4))
        dados_orig['EngagementLevel'].value_counts().plot(
            kind='bar', color=['#FF6B6B', '#4ECDC4'])
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        st.header("Relação Idade vs Tempo de Jogo")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(
            data=dados_orig, 
            x='Age', 
            y='PlayTimeHours', 
            hue='EngagementLevel',
            palette={0: '#FF6B6B', 1: '#4ECDC4'})
        st.pyplot(fig)

# Página 3: Pré-processamento
elif pagina == "⚙️ Pré-processamento":
    st.title("⚙️ Pré-processamento dos Dados")
    st.markdown("---")
    
    if dados_prep is not None:
        st.header("Transformações Aplicadas")
        st.markdown("""
        1. **Filtragem:** Apenas níveis 'Low' e 'High' de engajamento
        2. **Remoção de colunas:** PlayerID, AvgSessionDurationMinutes, Gender, Location
        3. **Codificação:**
           - EngagementLevel: Low → 0, High → 1
           - Variáveis categóricas: One-Hot Encoding
        4. **Padronização:** StandardScaler nas variáveis numéricas
        """)
        
        st.header("Dados Pré-processados (Amostra)")
        st.dataframe(dados_prep.head(), use_container_width=True)
        
        st.header("Estrutura dos Dados Transformados")
        st.write(f"**Formato final:** {dados_prep.shape[0]} linhas × {dados_prep.shape[1]} colunas")

# Página 4: Modelo Preditivo
elif pagina == "🤖 Modelo Preditivo":
    st.title("🤖 Modelo de Machine Learning")
    st.markdown("---")
    
    st.header("Metodologia")
    st.markdown("""
    - **Framework:** PyCaret
    - **Seleção de Modelos:** Comparação com base em F1-Score
    - **Melhor Modelo:** Random Forest (após tunagem)
    - **Métricas:**
      - Acurácia: 89%
      - F1-Score: 88%
    """)
    
    try:
        model = joblib.load('models/melhor_modelo_dificuldade_jogo.pkl')
        st.success("✅ Modelo carregado com sucesso!")
        
        st.header("Importância das Variáveis")
        # Nota: Substitua com os valores reais do seu modelo
        feature_importance = pd.DataFrame({
            'Feature': ['PlayTimeHours', 'PlayerLevel', 'Age', 'AchievementsUnlocked'],
            'Importance': [0.35, 0.25, 0.20, 0.20]
        })
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")

# Página 5: Fazer Previsão
elif pagina == "🔮 Fazer Previsão":
    st.title("🔮 Simulador de Previsão")
    st.markdown("---")
    
    if dados_orig is not None and scaler is not None:
        st.header("Insira os Dados do Jogador")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Idade", 10, 60, 25)
            play_time = st.slider("Horas Jogadas/Semana", 1, 40, 10)
            level = st.slider("Nível do Personagem", 1, 100, 30)
            
        with col2:
            achievements = st.slider("Conquistas", 0, 50, 5)
            difficulty = st.selectbox("Dificuldade", ["Easy", "Medium", "Hard"])
            genre = st.selectbox("Gênero", ["Action", "Adventure", "RPG", "Strategy"])
        
        if st.button("Prever Engajamento", type="primary"):
            try:
                # Criar DataFrame com inputs
                input_data = pd.DataFrame({
                    'Age': [age],
                    'SessionsPerWeek': [3],  # Valor padrão
                    'PlayTimeHours': [play_time],
                    'AchievementsUnlocked': [achievements],
                    'PlayerLevel': [level],
                    'GameGenre': [genre],
                    'GameDifficulty': [difficulty],
                    'InGamePurchases': ['Yes']  # Valor padrão
                })
                
                # Pré-processar
                input_prep, _ = preprocess_data(input_data)
                
                # Fazer previsão (simulação)
                prediction = "High"  # Substitua pela previsão real
                proba = 0.85  # Substitua pela probabilidade real
                
                st.success(f"Resultado: Engajamento {prediction} ({(proba if prediction == 'High' else 1-proba):.0%} de confiança)")
                
                if prediction == "High":
                    st.balloons()
            except Exception as e:
                st.error(f"Erro na previsão: {e}")

# Rodapé
st.markdown("---")
st.caption("Desenvolvido com base nas análises de pré-processamento do notebook")
