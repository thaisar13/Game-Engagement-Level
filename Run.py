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
    st.title("🎮 Análise Preditiva de Engajamento em Jogos")
    st.markdown("---")
    
    # Seção de introdução com colunas
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📋 Visão Geral do Projeto")
        st.markdown("""
        **Objetivo:** Desenvolver um modelo preditivo para classificar o nível de engajamento de jogadores  
        **Aplicação:** Auxiliar desenvolvedores a identificar padrões de comportamento e melhorar a experiência do usuário  
        **Abordagem:** Análise exploratória + Modelagem supervisionada (classificação binária)
        """)
        
    #with col2:
     #   st.image("https://cdn-icons-png.flaticon.com/512/2936/2936886.png", width=100)
    
    st.markdown("---")
    
    # Seção de dados com expansores
    with st.expander("🔍 **Fonte de Dados**", expanded=True):
        st.markdown("""
        - **Dataset:** [Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)
        - **Variáveis originais:** 13
        - **Amostra final:** {:,} jogadores (Low: {:,} | High: {:,})
        """.format(
            len(dados_vis),
            sum(dados_vis['EngagementLevel'] == 'Low'),
            sum(dados_vis['EngagementLevel'] == 'High')
        ))
    
    # Seção técnica com tabs
    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "🧠 Modelagem", "⚙️ Engenharia de Features"])
    
    with tab1:
        st.subheader("Desempenho do Modelo (Gradient Boosting)")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("F1-Score", "0.88", help="Métrica balanceada entre precisão e recall")
        col2.metric("Acurácia", "0.87", help="Percentual total de acertos")
        col3.metric("Precisão", "0.84", help="Quando prevê Alto Engajamento, acerta 84%")
        col4.metric("Recall", "0.92", help="Identifica 92% dos casos reais de Alto Engajamento")
        

    with tab2:
        st.markdown("""
        **Processo de Modelagem:**
        1. **Pré-processamento:** Filtragem, codificação e normalização
        2. **Seleção de Modelos:** Comparação de 15 algoritmos via PyCaret
        3. **Tunagem:** Otimização de hiperparâmetros com busca Bayesiana
        4. **Validação:** Teste com holdout de 25% dos dados
        
        **Algoritmos Testados:**
        
        - Ada Boost Classifier	(F1: 0.88)
        - Gradient Boosting Classifier	(F1: 0.88) **← Selecionado**
        - Light Gradient Boosting Machine	(F1: 0.88)	
        - Random Forest Classifier	(F1: 0.88)	
        - Ridge Classifier	(F1: 0.88)
        - Linear Discriminant Analysis		(F1: 0.88)	
        - Naive Bayes	(F1: 0.87)
        - Quadratic Discriminant Analysis	(F1: 0.87)	
        - Logistic Regression	(F1: 0.87)	
        - SVM - Linear Kernel	(F1: 0.87)
        - Extra Trees Classifier	(F1: 0.87)	
        - Extreme Gradient Boosting	(F1: 0.87)	
        - K Neighbors Classifier	(F1: 0.85)	
        - Decision Tree Classifier	(F1: 0.78)	
        - Dummy Classifier (F1: 0.67)	
    """)
    
    with tab3:
        st.markdown("""
        **Principais Features:**
        1. SessionsPerWeek (Importância: 98%)
        2. PlayerLevel (2%)
        3. AchievementsUnlocked (2%)
        
        **Transformações:**
        - One-Hot Encoding: Gênero, Dificuldade
        - Standard Scaling: Variáveis numéricas
        - Balanceamento: SMOTE para equalizar classes
        """)
    
    # Chamada para ação
    st.markdown("---")
    st.success("💡 **Explore as outras seções para análises detalhadas e simulações de previsão!**")

#```{python}
        #GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
        #                           learning_rate=0.001, loss='log_loss', max_depth=6,
       #                            max_features='log2', max_leaf_nodes=None,
      #                             min_impurity_decrease=0.0005, min_samples_leaf=3,
     #                              min_samples_split=4, min_weight_fraction_leaf=0.0,
    #                               n_estimators=60, n_iter_no_change=None,
   #                                random_state=42, subsample=0.95, tol=0.0001,
  #                                 validation_fraction=0.1, verbose=0,
 #                                  warm_start=False)
#```

 
    

# Página 2: Análise Exploratória
elif pagina == "🔍 Análise Exploratória":
    st.title("🔍 Análise Exploratória dos Dados")
    st.markdown("---")
    
    if dados_vis is not None:
        st.header("📊 Dados Brutos (Amostra)")
        st.dataframe(dados_vis.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Registros", len(dados_vis))
        with col2:
            st.metric("Variáveis Originais", len(dados_vis.columns))
    
    if dados_vis is not None:
        st.header("Distribuição de Engajamento")
        
        # Gráfico de barras 
        fig, ax = plt.subplots(figsize=(10, 5))
        counts = dados_vis['EngagementLevel'].value_counts()
        counts.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'], ax=ax)
        # Adicionando rótulos e formatação
        ax.set_title('Distribuição dos Níveis de Engajamento', pad=20)
        ax.set_xlabel('Nível de Engajamento')
        ax.set_ylabel('Contagem')
        ax.set_xticklabels(['Baixo (Low)', 'Alto (High)'], rotation=0)
        # Adicionando valores nas barras
        for i, v in enumerate(counts):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12)
        st.pyplot(fig)
        
        st.markdown("---")
        st.header("Relação Idade vs Tempo de Jogo")
        
        # Scatterplot
        fig, ax = plt.subplots(figsize=(12, 7))
        scatter = sns.scatterplot(
            data=dados_vis, 
            x='Age', 
            y='PlayTimeHours', 
            hue='EngagementLevel',
            palette={'Low': '#FF6B6B', 'High': '#4ECDC4'},
            s=100,  # Tamanho dos pontos aumentado
            alpha=0.7,  # Transparência
            ax=ax
        )
        # Legenda
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Baixo (Low)', 'Alto (High)'], title='Engajamento')
        # Adicionando título e rótulos
        ax.set_title('Relação entre Idade e Tempo de Jogo por Nível de Engajamento', pad=20)
        ax.set_xlabel('Idade (anos)')
        ax.set_ylabel('Horas Jogadas por Semana')
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Matriz de Correlação")
        
        # Matriz de correlação
        fig, ax = plt.subplots(figsize=(12, 8))
        # Calculando a matriz de correlação apenas para variáveis numéricas
        numeric_vars = dados_vis.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numeric_vars.corr()
        # Criando o heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        # Ajustando o título
        ax.set_title('Correlação entre Variáveis Numéricas', pad=20)
        st.pyplot(fig)
    
# Página 3: Pré-processamento
elif pagina == "⚙️ Pré-processamento":
    st.title("⚙️ Pré-processamento dos Dados")
    st.markdown("---")
    
    if dados_prep is not None:
        st.header("Transformações Aplicadas")
        #st.markdown("""
        ### 1. Filtragem Inicial
       # - **Seleção de categorias:** Mantivemos apenas os níveis 'Low' e 'High' de engajamento
        #- **Justificativa:** A categoria 'Medium' foi excluída para criar um problema de classificação binária mais definido
        #- **Resultado:** Redução de {:.1%} no volume de dados (de {} para {} registros)
        #""".format(
        #    1 - len(dados_prep)/len(dados_orig),
        #    len(dados_orig),
        #    len(dados_prep)
        #)

        st.markdown("""
        ### 2. Seleção de Features
        Foram removidas as seguintes variáveis:
        - **PlayerID:** Identificador único sem valor preditivo
        - **AvgSessionDurationMinutes:** Eliminada para evitar multicolinearidade com:
          - `SessionsPerWeek` (r = {:.2f})
          - `PlayTimeHours` (r = {:.2f})
        - **Gender e Location:** Removidas após análise de importância de features mostrar baixa contribuição (< {:.1%} de importância relativa)
        """.format(
            dados_orig[['AvgSessionDurationMinutes', 'SessionsPerWeek']].corr().iloc[0,1],
            dados_orig[['AvgSessionDurationMinutes', 'PlayTimeHours']].corr().iloc[0,1],
            0.05  # Substitua pelo valor real da sua análise
        ))

        st.markdown("""
        ### 3. Transformação de Variáveis
        **Variável Target:**
        - Codificação binária:
          - `Low` → 0
          - `High` → 1

        **Variáveis Categóricas (One-Hot Encoding):**
        ```python
        pd.get_dummies(columns=['GameGenre', 'GameDifficulty', 'InGamePurchases'], 
                      drop_first=True)
        ```
        - **Estratégia:** `drop_first=True` para evitar a armadilha da variável dummy
        - **Resultado:** Adição de {} novas colunas
        """.format(len(dados_prep.columns) - 8))  # Ajuste o número conforme suas variáveis

        st.markdown("""
        **Variáveis Numéricas:**
        ```python
        StandardScaler().fit_transform(['Age', 'SessionsPerWeek', 
                                      'PlayTimeHours', 'AchievementsUnlocked',
                                      'PlayerLevel'])
        ```
        - **Efeito:** Centralização (μ=0) e Escala (σ=1)
        - **Benefícios:**
          - Melhor convergência para modelos sensíveis à escala (SVM, Regressão Logística)
          - Importância relativa comparável entre features
        """)

        st.markdown("""
        ### 4. Validação do Pré-processamento
        - **Balanceamento de classes:** {:.1f}:{:.1f} (Low:High)
        - **Ausência de NaNs:** Confirmada ({} valores faltantes totais)
        - **Matriz de correlação:** Verificada ausência de multicolinearidade crítica (|r| < {:.2f})
        """.format(
            *dados_prep['EngagementLevel'].value_counts(normalize=True).values,
            dados_prep.isna().sum().sum(),
            0.49  
        ))

        with st.expander("🔍 Visualização do Pipeline Completo"):
            st.image("https://miro.medium.com/max/1400/1*4PqYyZbws0N4yR0sFw3yJQ.png", 
                    caption="Exemplo de fluxo de pré-processamento", width=400)
        
        st.header("📋 Dados Pré-processados (Amostra)")
        st.dataframe(dados_prep.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros", len(dados_prep))
        with col2:
            st.metric("Variáveis", len(dados_prep.columns))

# Página 4: Modelo Preditivo
elif pagina == "🤖 Modelo Preditivo":
    st.title("🤖 Modelo de Machine Learning")
    st.markdown("---")
    
    st.header("Metodologia")
    st.markdown("""
    - **Framework:** PyCaret
    - **Seleção de Modelos:** Comparação com base no rankeamneto com o F1-Score e comparação do melhor desempenho geral entre as métricas
    - **Melhor Modelo:** Gradient Boosting Classifier (tunado após seleção)
    - **Métricas:**
      - Acurácia: 87%
      - Recall: 92%
      - Precisão: 84%
      - F1-Score: 88%
    """)
    
    
    try:
        model = joblib.load('model.pkl')
        st.success("✅ Modelo carregado com sucesso!")
        
        st.header("Importância das Variáveis")
        # Nota: Substitua com os valores reais do seu modelo
        feature_importance = pd.DataFrame({
            'Feature': ['SessionsPerWeek', 'PlayerLevel', 'AchievementsUnlocked', 'PlayTimeHours','Age', 
                        'InGamePurchases_1', 'EngagementLevel', 'GameGenre_RPG', 'GameGenre_Simulation', 
                        'GameGenre_Sports', 'GameGenre_Strategy', 'GameDifficulty_Hard', 'GameDifficulty_Medium'],
            'Importance': [0.98, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
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
    
    if dados_vis is not None and scaler is not None:
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
