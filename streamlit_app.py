# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px  


# Configurações da página
st.set_page_config(
    page_title="Análise Preditiva de Engajamento dos Jogadores",
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
    cat_vars = ['Gender', 'GameDifficulty']
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
                        'GameDifficulty', 'Gender']
        
        dados_vis = dados[cols_to_keep].copy()
        
        # Processar dados para modelagem
        dados_prep, scaler = preprocess_data(dados_vis)
        
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
        st.subheader("Desempenho do Modelo (Ada Boost Classifier)")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("F1-Score", "0.88", help="Métrica balanceada entre precisão e recall")
        col2.metric("Acurácia", "0.87", help="Percentual total de acertos")
        col3.metric("Precisão", "0.83", help="Quando prevê Alto Engajamento, acerta 84%")
        col4.metric("Sencibilidade", "0.93", help="Identifica 92% dos casos reais de Alto Engajamento")
          
    with tab2:
        st.markdown("""
        **Processo de Modelagem:**
        1. **Pré-processamento:** Filtragem, codificação e normalização
        2. **Seleção de Modelos:** Comparação de 15 algoritmos via PyCaret (seleciondo o Ada Boos Classifier)
        3. **Tunagem:** Otimização de hiperparâmetros para melhor F1-Score
        4. **Validação:** Teste com holdout de 25% dos dados
        
        **Algoritmos Testados:**""")

        # Dados da tabela
        data = {
            "Model": [
                "Ada Boost Classifier",
                "Gradient Boosting Classifier",
                "Light Gradient Boosting Machine",
                "Ridge Classifier",
                "Linear Discriminant Analysis",
                "Random Forest Classifier",
                "Naive Bayes",
                "Quadratic Discriminant Analysis",
                "Logistic Regression",
                "Extra Trees Classifier",
                "SVM - Linear Kernel",
                "Extreme Gradient Boosting",
                "K Neighbors Classifier",
                "Decision Tree Classifier",
                "Dummy Classifier"
            ],
            "F1-Score": [0.8792, 0.8776, 0.8767, 0.8759, 0.8759, 0.8755, 0.8743, 0.8735, 0.8723, 0.8705, 0.8680, 0.8679, 0.8540, 0.7823, 0.6669],
            "Acurácia": [0.8719, 0.8711, 0.8707, 0.8716, 0.8716, 0.8696, 0.8703, 0.8696, 0.8689, 0.8651, 0.8663, 0.8628, 0.8491, 0.7840, 0.5003],
            "Precisão": [0.8324, 0.8365, 0.8381, 0.8478, 0.8478, 0.8382, 0.8491, 0.8485, 0.8510, 0.8378, 0.8397, 0.8373, 0.8278, 0.7866, 0.5003],
            "Sensibilidade": [0.9316, 0.9230, 0.9192, 0.9060, 0.9060, 0.9164, 0.9011, 0.9002, 0.8947, 0.9058, 0.8985, 0.9011, 0.8821, 0.7799, 1.0000]
        }

        df = pd.DataFrame(data)
        
        st.write("Tabela comparativa de desempenho (sem AUC, Kappa, MCC e Tempo de Treinamento)")
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <style>
            .dataframe td {
                text-align: center !important;
            }
            .dataframe th {
                text-align: center !important;
            }
        </style>
        """, unsafe_allow_html=True)
            
            
    with tab3:
        st.markdown("""
        **Váriaveis Descartadas:**
        1. PlayerID
        2. AvgSessionDurationMinutes
        3. GameGenre
        4. InGamePurchases
        5. Location

        **Principais Features:**
        1. SessionsPerWeek (Importância: 60%)
        2. PlayTimeHours (12%)
        3. PlayerLevel (10%)
        4. AchievementsUnlocked (10%)
        
        **Transformações:**
        - One-Hot Encoding nas variáveis categóricas
        - Standard Scaling na variáveis numéricas
        """)
    
    # Chamada para ação
    st.markdown("---")
    st.success("💡 **Explore as outras seções para análises detalhadas e simulações de previsão!**")


# Página 2: Análise Exploratória
elif pagina == "🔍 Análise Exploratória":
    st.title("🔍 Análise Exploratória dos Dados")
    st.markdown("---")
    
    if dados_vis is not None:
        st.header("🎲 Dados Brutos (Amostra)")
        st.dataframe(dados_vis.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Registros", len(dados_vis))
        with col2:
            st.metric("Variáveis Originais", len(dados_vis.columns))
    
    if dados_vis is not None:

        st.header("📋 Análise das Variáveis")
        # Gráfico Univariado
        st.sidebar.subheader("Gráfico Univariado")
        var_univariada = st.sidebar.selectbox(
            "Selecione a variável para análise univariada:",
            options=['EngagementLevel', 'Age', 'PlayTimeHours', 'SessionsPerWeek', 'Gender',
                     'PlayerLevel', 'AchievementsUnlocked', 'GameDifficulty'],
            index=0
        )
        
        # Gráfico Bivariado
        st.sidebar.subheader("Gráfico Bivariado")
        var_x = st.sidebar.selectbox(
            "Selecione a variável para o eixo X:",
            options=['Age', 'PlayTimeHours', 'SessionsPerWeek', 'Gender',
                     'PlayerLevel', 'AchievementsUnlocked', 'GameDifficulty'],
            index=0
        )
        
        var_y = st.sidebar.selectbox(
            "Selecione a variável para o eixo Y:",
            options=['Age', 'PlayTimeHours', 'SessionsPerWeek', 'Gender',
                     'PlayerLevel', 'AchievementsUnlocked', 'GameDifficulty'],
            index=1
        )
        
        # Layout dos gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Distribuição de {var_univariada} por Engajamento")
            
            # Usando Plotly para interatividade
            fig = px.histogram(dados_vis, x=var_univariada, color="EngagementLevel",
                               nbins=30, barmode="overlay",
                               title=f"Distribuição de {var_univariada}",
                               opacity=0.7,
                               color_discrete_map={
                                  "High": '#4ECDC4',
                                  "Low": '#FF6B6B'
                              })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(f"Relação entre {var_x} e {var_y} por Engajamento")
            
            # Gráfico de dispersão com Plotly
            fig2 = px.scatter(dados_vis, x=var_x, y=var_y, color="EngagementLevel",
                              title=f"{var_x} vs {var_y}",
                              opacity=0.6,
                              color_discrete_map={
                                  "High": '#4ECDC4',
                                  "Low": '#FF6B6B'
                              })
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Como usar esta análise?"):
            
            st.markdown("""
            - No gráfico à esquerda: Compare como cada nível de engajamento se distribui para uma variável específica
            - No gráfico à direita: Explore relações entre pares de variáveis numéricas
            - Passe o mouse sobre os pontos para ver detalhes
            - Use os menus laterais para selecionar diferentes variáveis
            """)
        
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

        st.markdown("---")
        st.header("🧐 Interpretações Gerais")
        
        st.markdown("""
        ### 📊 Análise das Variáveis Categóricas
        
        - Por exceção das variável 'EngagementLevel', as variáveis categóricas restantes ('GameDifficulty' e Gender')
        apresentam algum desbalanceamento entre suas categorias.
        - Todas as variáveis categóricas apresentam uma proporção equilibrada considerando as classe de alto e baixo engajamento.
        - **Implicação**: 
          - Nenhuma categoria domina excessivamente os entre as classes avaliadas o que pode dificultar identificação dos 
          padrões de engajamento
        """)
        
        st.markdown("""
        ### 📈 Análise das Variáveis Contínuas
        
        - Por exceção das variável 'SessionsPerWeek', as variáveis quantitativas restantes ('Age', 'PlayTimeHours', 
        'PlayerLevel' e 'AchievementsUnlocked') apresentam aproximadamente uma distribuição uniforme.
        - Considerando as calsse de alto e baixo engajamento, todas as variáveis apresentam quase a mesma frequência de resposta. 
        - **Implicação**: 
          - A variável 'SessionsPerWeek' é provavelmente a variável mais importante para a classificação do engajamento do jogador.

        """)
        st.markdown("""
        ### 📉 Análise Bivariada

        - De modo geral as variáveis não apresentam uma relação definida, apresentando "nuvens" de pontos sem um padrão
        específico.
        - **Implicação**: 
          - Relações lineares aparentemente ausentes
          - Necessidade de investigar possíveis padrões não-lineares
        """)
        
        st.markdown("""
        ### 🔢 Análise de Correlação Numérica
        
        - Correlações geralmente próximas de zero
        - Ausência de relações lineares fortes entre features
        - **Implicação**:
          - Desafio para modelos lineares tradicionais
          - Oportunidade para algoritmos que capturam relações complexas
        """)
        st.markdown("---")
        st.header("🧐 Interpretações Gerais")
        
        st.markdown("""
        ### 📊 Análise das Variáveis Categóricas
        
        Diferentemente da variável 'EngagementLevel' as variáveis categóricas 'GameDifficulty' e 'Gender' apresentam 
        um certo desbalanceamento entre 
        suas categorias. Curiosamente, quando observamos a distribuição dessas variáveis em relação às classes de alto 
        e baixo engajamento, nota-se uma proporção equilibrada entre os grupos. Esta característica sugere que 
        nenhuma categoria específica domina excessivamente qualquer classe de engajamento, o que pode tornar mais 
        desafiador a identificação de padrões categóricos claros para a classificação. 
        """)
        
        st.markdown("""
        ### 📈 Análise das Variáveis Contínuas
        
        Dentre as variáveis quantitativas analisadas, 'SessionsPerWeek' se destaca por apresentar uma distribuição 
        distinta das demais ('Age', 'PlayTimeHours', 'PlayerLevel' e 'AchievementsUnlocked'), que seguem 
        aproximadamente uma distribuição uniforme. Quando examinamos o comportamento dessas variáveis entre as 
        classes de engajamento, observamos frequências de resposta muito similares. Esta análise sugere que 
        'SessionsPerWeek' provavelmente será a variável mais discriminativa e importante para a classificação do 
        nível de engajamento dos jogadores.
        """)
        
        st.markdown("""
        ### 📉 Análise Bivariada
        
        A exploração das relações entre pares de variáveis revela predominantemente padrões difusos, com formações de 
        "nuvens" de pontos sem geometria definida. Esta ausência de padrões lineares claros entre as variáveis indica 
        que possíveis relações existentes provavelmente seguem padrões mais complexos e não-lineares, que não são 
        facilmente identificáveis através de análise visual simples.
        """)
        
        st.markdown("""
        ### 🔢 Análise de Correlação Numérica
        
        Os coeficientes de correlação calculados entre as variáveis numéricas apresentam valores geralmente próximos 
        de zero, confirmando a ausência de relações lineares fortes entre os atributos. Este cenário representa um 
        desafio particular para modelos que dependem fundamentalmente de relações lineares, mas ao mesmo tempo abre 
        oportunidades para a aplicação de algoritmos mais sofisticados capazes de capturar interações e padrões 
        não-lineares nos dados.
        """)
        
# Página 3: Pré-processamento
elif pagina == "⚙️ Pré-processamento":
    st.title("⚙️ Pré-processamento dos Dados")
    st.markdown("---")
    
    if dados_prep is not None:
        st.header("🛠️ Transformações Aplicadas")
      
        st.markdown("""
        ### 1. Seleção de Features
        Foram removidas as seguintes variáveis:
        - **'PlayerID':** Identificador único sem valor preditivo
        - **'AvgSessionDurationMinutes':** Eliminada para evitar multicolinearidade com `SessionsPerWeek` e `PlayTimeHours`
        - **'GameGenre', 'InGamePurchases' e 'Location':** Removidas após análise de importância de features por não apresentarem relevância nos melhores modelos testados
        """)

        st.markdown("""
        ### 2. Transformação de Variáveis
        **Variável Target:**
        - Codificação binária:
          - `Low` → 0
          - `High` → 1

        **Variáveis Categóricas (One-Hot Encoding):**
        ```python
        pd.get_dummies(columns=['GameDifficulty', 'Gender'], drop_first=True)
        ```
        - **Resultado:** Adição de {} novas colunas
        """.format(len(dados_prep.columns)))  # Ajuste o número conforme suas variáveis
        #- **Estratégia:** `drop_first=True` para evitar a armadilha da variável dummy

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
        ### 3. Validação do Pré-processamento
        - **Balanceamento de classes:** {:.1f}:{:.1f} (Low:High)
        - **Ausência de NaNs:** Confirmada ({} valores faltantes totais)
        """.format(
            *dados_prep['EngagementLevel'].value_counts(normalize=True).values,
            dados_prep.isna().sum().sum(),  
        ))

#        with st.expander("🔍 Visualização do Pipeline Completo"):
 #           st.image("https://miro.medium.com/max/1400/1*4PqYyZbws0N4yR0sFw3yJQ.png", 
  #                  caption="Exemplo de fluxo de pré-processamento", width=400)
        
        st.header("🎲 Dados Pré-processados (Amostra)")
        st.dataframe(dados_prep.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros", len(dados_prep))
        with col2:
            st.metric("Variáveis", len(dados_prep.columns))

# Página 4: Modelo Preditivo
elif pagina == "🤖 Modelo Preditivo":
    st.title("🤖 Modelo Preditivo: Ada Boost Classifier")
    st.markdown("---")

    st.header("Metodologia")
    st.markdown("""
    - **Framework:** PyCaret
    - **Seleção de Modelos:** Comparação com base no rankeamneto com o F1-Score, seguido do melhor desempenho geral entre as métricas
    - **Melhor Modelo:** Ada Boost Classifier (tunado após seleção)
    - **Métricas:**
      - Acurácia: 87%
      - Sencibilidade: 93%
      - Precisão: 83%
      - F1-Score: 88%
      - AUC: 92%
    """)

    tab1, tab2 = st.tabs(["🔍 Interpretação do Modelo", "🎯 Quem é o Ada Boost Classifier?"])
    
    with tab1:
        st.header("🔍 Interpretação do Modelo")
        try:
            pipeline = joblib.load('modelo.pkl')
            #st.success("✅ Modelo carregado com sucesso!")
            
            st.markdown("""
            ### Significado das Métricas
            
            **Interpretação:**  
            
            O modelo demonstra um **bom desempenho geral** (Acurácia de 87%), com destaque para sua capacidade de:  
            
            - **Capturar casos Positivos**: Alta Sencibilidade (93%) indica que o modelo identifica efetivamente jogadores engajados (apenas 8% de falsos negativos)  
            - **Distinguir Classes**: AUC (Área sib a Curva ROC) de 92% revela excelente separação entre os jogadores de baixo engajamento dos de alto engajamento 
            - **Equilíbrio**: F1-Score (88%) mostra boa harmonia entre Precisão e Recall  
            - **Chance de Erro**: Precisão (83%) sugere que, quando o modelo prevê "alto engajamento", há 16% de chance de ser falso positivo. Algo espera dada a 
            sobreposição natural nos padrões de engajamento e das variáveis preditoras apresentarem um poder limitado de discriminação  
            """)
            
            st.markdown(""" ### Importância das Variáveis""")

            feature_importance = pd.DataFrame({
                #'Feature': ['SessionsPerWeek', 'PlayerLevel', 'AchievementsUnlocked', 'PlayTimeHours','Age', 
                 #           'InGamePurchases_1', 'EngagementLevel', 'GameGenre_RPG', 'GameGenre_Simulation', 
                  #          'GameGenre_Sports', 'GameGenre_Strategy', 'GameDifficulty_Hard', 'GameDifficulty_Medium'],
                #'Importance': [0.98, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                'Variável': ['Age',
                            'PlayTimeHours',
                            'SessionsPerWeek',
                            'PlayerLevel',
                            'AchievementsUnlocked',
                            'GameDifficulty_Hard',
                            'GameDifficulty_Medium',
                            'Gender_Male'],
                'Importância': [ 0.06,
                                0.12,
                                0.6,
                                0.1,
                                0.1,
                                0,
                                0,
                                0.02]

            })
            #feature_importance = pd.DataFrame({'Variavel': pipeline.named_steps['actual_estimator'].feature_names_in_, 
            #                                   'Importancia': pipeline.named_steps['actual_estimator'].feature_importances_})

            #st.write("Classes do modelo:", pipeline.classes_)
            #st.write("Feature names:", pipeline.named_steps['actual_estimator'].feature_names_in_)
            #st.write(pipeline.named_steps)
            #st.write("Importância das Features:", pipeline.named_steps['actual_estimator'].feature_importances_)

            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=feature_importance, x='Importância', y='Variável', palette='viridis')
            st.pyplot(fig)
            
            st.markdown(""" 
            A análise de importância de variáveis revela que 'SessionsPerWeek' é o preditor mais relevante, respondendo por 60% do poder explicativo do modelo, 
            indicando que a frequência de sessões é o fator determinante para classificar o engajamento. As demais variáveis ('Age', 'PlayTimeHours', 
            'PlayerLevel', 'AchievementsUnlocked' e 'Gender_Male') compartilham os 40% restantes de importância, com 'Gender_Male' sendo a menos influente. 
            Embora as dummies 'GameDifficulty_Hard' e 'GameDifficulty_Medium' não apresentem contribuição significativa individualmente, 
            sua inclusão se mostrou necessária, pois a remoção da variável 'GameDifficulty' causou superconcentração de importância (100%) em 'SessionsPerWeek', 
            sugerendo que essas categorias atuam como reguladores da importância da variável principal, garantindo uma distribuição mais equilibrada do poder 
            preditivo entre as features.
            """)
    
            
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")

    with tab2:
        # Seção 3: Conhecendo o Gradient Boosting
        st.header("🎯 Quem é o Ada Boost Classifier?")
        st.markdown("""
        <div style="text-align: justify">
        O <strong>Gradient Boosting Classifier</strong> é como um time de especialistas trabalhando em equipe, onde cada novo membro 
        aprende com os erros dos anteriores. Veja como ele se destaca:
        </div>
        """, unsafe_allow_html=True)
        # Explicação visual em colunas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### 🧠 Como Funciona?
            
            ##### 🌱 **Passo Inicial - Base Simples**
            - Começa com um modelo fraco (ex: árvore de decisão rasa - stump)
            - Todos os exemplos têm peso igual inicialmente
            
            ##### 🔄 **Processo Iterativo - Aprendizado com Erros**
            1. **Primeira Iteração**:
               - O stump faz predições iniciais
               - Erros são identificados e os exemplos mal classificados recebem mais peso
            
            2. **Iterações Seguintes**:
               - Cada novo stump foca nos exemplos mais difíceis (com maior peso)
               - Modelos subsequentes "herdam" os erros corrigidos anteriormente
            
            ##### ⚖️ **Mecanismo de Peso**
            - **Peso dos Exemplos**: Aumenta para casos mal classificados
            - **Peso dos Modelos**: Stumps mais precisos têm maior influência no voto final
            
            ##### ✨ **Resultado Final - Voto Ponderado**
            - Combina todas as previsões dos stumps
            - Cada contribuição é ponderada pela precisão do modelo
            
            ##### 🌟 **Vantagens Chave**
            - Foco automático nos casos mais difíceis
            - Simples e eficaz para problemas binários
            - Menos propenso a overfitting que algoritmos complexos
            
            """)
        with col2:

            st.markdown("""
            ### 🏆 Critério de Seleção do Modelo
            
            O modelo foi selecionado através de uma análise comparativa das métricas de desempenho, considerando inicialmente os **3 melhores modelos com base no F1-Score**. 
            Foi atribuído um sistema de pontuação por posição:
            
            - <b>1º lugar</b> em cada métrica: <b>+2 pontos</b>
            - <b>2º lugar</b>: <b>+1 ponto</b>
                        
            Após essa avaliação, os três melhores modelos ficaram com:
            
            <div style="margin-left: 20px;">
            
            <b>1. Ada Boost Classifier: (7 ponttos)</b>  
               - 🥇 <b>Melhor desempenho</b> em:  
                 • F1-Score
                 • Acurácia
                 • Sencibilidade  
               - 🥈 <b>Segunda melhor</b> em:
                 • Área sob a Curva ROC
            
            <b>2. Gradient Boosting Classifier (6 pontos)</b>  
               - 🥇 <b>Melhor desempenho</b> em:  
                 • Área sob a Curva ROC
               - 🥈 <b>Segundo melhor</b> em:  
                 • F1-Score
                 • Acurácia
                 • Sencibilidade
                 • Precisão
                 
            <b>3. Light Gradient Boosting Machine (2 pontos)</b>  
               - 🥇 <b>Melhor desempenho</b> em:  
                 • Precisão

            </div>
            
            <h4>Critério de Seleção Final</h4>
            Outro fator que influenciou na escolha do AdaBoost Classifier foi a distribuição de importância das variáveis. No Gradient Boosting Classifier, 
            a variável 'SessionsPerWeek' apresentava 97% de importância, reduzindo 'PlayTimeHour' a uma relevância quase nula - um padrão inadequado, pois: 
            um jogador pouco engajado pode ter várias sessões semanais mas com poucas horas jogadas em cada, enquanto um jogador altamente engajado pode 
            acumular muitas horas de jogo em poucas sessões prolongadas. O AdaBoost, ao distribuir melhor essa importância, captura essa nuance comportamental 
            de forma mais equilibrada.

            
            """, unsafe_allow_html=True)
            st.info(" **Observação Final:💡As diferenças entre as métricas dos dois modelos são muito sutis, não havendo um desempenho significativamente superior de um em relação ao outro.**")
                
        # Detalhes técnicos com expansor
        with st.expander("🧮 A Matemática por Trás", expanded=False):
            st.markdown("""
            **Fórmula de Atualização de Pesos**:
            ```
            w_i ← w_i * exp(α_t * I(y_i ≠ h_t(x_i)))
            ```
            Onde:
            - `α_t`: Peso do classificador (baseado em sua precisão)
            - `h_t(x_i)`: Predição do stump no passo t
            - `I()`: Função indicadora (1 se erro, 0 se correto)
            
            **Passo a Passo**:
            1. Inicializa pesos uniformes para todos os exemplos
            2. Para cada iteração:
               a. Treina stump nos dados com pesos atuais
               b. Calcula erro ponderado
               c. Atualiza pesos dos exemplos
               d. Atribui peso ao modelo baseado em sua precisão
            3. Combina todos os stumps via voto ponderado
            """)
            
        with st.expander("🔧 Configuração Técnica Detalhada", expanded=False):
            st.code("""
            AdaBoostClassifier(
                algorithm='SAMME.R',       # Versão real do algoritmo AdaBoost
                base_estimator=None,        # Por padrão usa DecisionTree com max_depth=1 (stump)
                learning_rate=1.0,          # Taxa de aprendizado (contribuição de cada modelo)
                n_estimators=50,           # Número de stumps (modelos fracos)
                random_state=42             # Reprodutibilidade
            )
            """, language='python')
        #st.markdown("""
        #<div style="background-color: #2e4057; padding: 15px; border-radius: 5px; color: white;">
        #<strong>💡 Curiosidade Técnica:</strong> Nosso modelo final combina <strong style="color:#f4d35e">150 dessas árvores</strong>, 
        #cada uma com profundidade máxima 4 (para evitar overfitting), usando taxa de aprendizado de 0.1.
        #</div>
        #""", unsafe_allow_html=True)
# Página 5: Previsão com o Modelo
elif pagina == "🔮 Fazer Previsão":
    st.title("🔮 Simulador de Previsão de Engajamento")
    st.markdown("---")
    
    if dados_vis is not None and scaler is not None:
        st.header("📋 Insira os Dados do Jogador")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Idade", 15, 50, 23)
            play_time = st.slider("Horas Jogadas/Dia", 1, 12, 3)
            sessions = st.slider("Sessões por Semana", 1, 20, 13)
            level = st.slider("Nível do Personagem", 1, 99, 25)
            #location = st.radio("Localização do Jogador", ["Ásia", "USA", "Europa", "Outro"], horizontal=True)
        
        with col2:
            achievements = st.slider("Conquistas Desbloqueadas", 0, 100, 30)
            difficulty = st.selectbox("Dificuldade do Jogo", ["Fácil", "Médio", "Difícil"], index=1)
            #genre = st.selectbox("Gênero do Jogo", ["Acition", "RPG", "Simulation", "Sports", "Strategy"])
            #purchases = st.radio("Realizou Compras no Jogo", ["Sim", "Não"], horizontal=True)
            gender = st.radio("Gênero do Jogador", ["Feminino", "Masculino"], horizontal=True)        
           
        if st.button("🔍 Prever Nível de Engajamento", type="primary", use_container_width=True):
            try:                
                # 1. Carrega o pipeline
                pipeline = joblib.load('modelo.pkl')
                
                # 2. Prepara os dados JÁ CODIFICADOS como o modelo espera
                input_data = pd.DataFrame({
                    'Age': [age],
                    'PlayTimeHours': [play_time],
                    'SessionsPerWeek': [sessions],
                    'PlayerLevel': [level],
                    'AchievementsUnlocked': [achievements],
                    
                    # Variáveis categóricas JÁ CODIFICADAS (one-hot)
                    'GameDifficulty_Hard': [1 if difficulty == "Difícil" else 0],
                    'GameDifficulty_Medium': [1 if difficulty == "Médio" else 0],
                    #'GameGenre_RPG': [1 if genre == "RPG" else 0],
                    #'GameGenre_Simulation': [1 if genre == "Simulation" else 0],
                    #'GameGenre_Sports': [1 if genre == "Sports" else 0],
                    #'GameGenre_Strategy': [1 if genre == "Strategy" else 0],
                    #'InGamePurchases_1': [1 if purchases == "Sim" else 0],
                    'Gender_Male': [1 if gender == "Masculino" else 0]#,
                    #'Location_Europe': [1 if location == "Europa" else 0],
                    #'Location_Other': [1 if location == "Outro" else 0],
                    #'Location_USA': [1 if location == "USA" else 0]
                })
                
                # 3. Garante a ordem correta das colunas
                input_data = input_data[pipeline.named_steps['actual_estimator'].feature_names_in_]
                #st.write("Classes do modelo:", pipeline.classes_)
                #st.write("Feature names:", pipeline.named_steps['actual_estimator'].feature_names_in_)
                #st.write(pipeline.named_steps['actual_estimator'])
                #st.write("Importância das Features:", pipeline.named_steps['actual_estimator'].feature_importances_)

                try:
                    # 4. Faz a previsão (o imputer vai lidar com quaisquer valores faltantes)
                    prediction = pipeline.predict(input_data)[0]
                    proba = pipeline.predict_proba(input_data)[0][1]
                    
                    #st.success(f"Previsão: {prediction} (Probabilidade: {proba:.2%})")
                    
                except Exception as e:
                    st.error(f"Erro na previsão: {str(e)}")
                    st.write("Dados enviados:", input_data)
                    st.write("Features esperadas:", pipeline.named_steps['actual_estimator'].feature_names_in_)
                    st.write("Classes do modelo:", pipeline.classes_)
                    st.write("Feature names:", pipeline.named_steps['actual_estimator'].feature_names_in_)
                    st.write(pipeline.named_steps)
                
                # Exibir resultados
                st.markdown("---")
                st.subheader("🧞‍♂️ Resultado da Previsão")
                    
                # URLs das imagens PNG dos emojis (em alta resolução)
                emoji_urls = {
                    
                    "game": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f3ae/512.png",
                    "book": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f4d6/512.png",
                
                    "older_woman": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f475/512.png",
                    "girl": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f467/512.png",
                    "woman": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f469/512.png",
                  
                    "older_man": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f474/512.png",
                    "boy": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f466/512.png",
                    "man": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f468/512.png"
                }
                
                if prediction == 1:
                    st.success(f"## Alto Engajamento ({proba:.2%} de probabilidade)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(emoji_urls["game"], width=200)
                    with col2:
                        if gender == 'Feminino':
                            if age >= 40:
                                st.image(emoji_urls["older_woman"], width=200)
                            elif age <= 25:
                                st.image(emoji_urls["girl"], width=200)
                            else:
                                st.image(emoji_urls["woman"], width=200)
                        else:  # Masculino
                            if age >= 40:
                                st.image(emoji_urls["older_man"], width=200)
                            elif age <= 25:
                                st.image(emoji_urls["boy"], width=200)
                            else:
                                st.image(emoji_urls["man"], width=200)
                else:
                    st.warning(f"## Baixo Engajamento ({(1-proba):.2%} de probabilidade)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(emoji_urls["book"], width=200)
                    with col2:
                        if gender == 'Feminino':
                            if age >= 40:
                                st.image(emoji_urls["older_woman"], width=200)
                            elif age <= 25:
                                st.image(emoji_urls["girl"], width=200)
                            else:
                                st.image(emoji_urls["woman"], width=200)
                        else:  # Masculino
                            if age >= 40:
                                st.image(emoji_urls["older_man"], width=200)
                            elif age <= 25:
                                st.image(emoji_urls["boy"], width=200)
                            else:
                                st.image(emoji_urls["man"], width=200)
                        
            except Exception as e:
                st.error(f"Erro na previsão: {str(e)}")
                
# Rodapé
st.markdown("---")
st.caption("Desenvolvido com base nas análises de pré-processamento do notebook disponíveis no [GitHub](https://github.com/thaisar13/Game-Engagement-Level)")
