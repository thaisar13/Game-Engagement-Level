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
        2. **Seleção de Modelos:** Comparação de 15 algoritmos via PyCaret (seleciondo o Gradient Boosting)
        3. **Tunagem:** Otimização de hiperparâmetros para melhor F1-Score
        4. **Validação:** Teste com holdout de 25% dos dados
        
        **Algoritmos Testados:**""")

        # Dados da tabela
        data = {
            "Model": [
                "Gradient Boosting Classifier",
                "Ada Boost Classifier",
                "Light Gradient Boosting Machine",
                "Random Forest Classifier",
                "Ridge Classifier",
                "Linear Discriminant Analysis",
                "Naive Bayes",
                "Quadratic Discriminant Analysis",
                "Logistic Regression",
                "SVM - Linear Kernel",
                "Extra Trees Classifier",
                "Extreme Gradient Boosting",
                "K Neighbors Classifier",
                "Decision Tree Classifier",
                "Dummy Classifier"
            ],
            "F1-Score": [0.8784, 0.8792, 0.8772, 0.8765, 0.8760, 0.8760, 0.8748, 0.8745, 0.8734, 0.8721, 0.8714, 0.8677, 0.8548, 0.7823, 0.6669],
            "Acurácia": [0.8720, 0.8718, 0.8712, 0.8705, 0.8716, 0.8716, 0.8709, 0.8705, 0.8700, 0.8667, 0.8663, 0.8624, 0.8499, 0.7826, 0.5003],
            "Precisão": [0.8371, 0.8323, 0.8390, 0.8382, 0.8477, 0.8477, 0.8493, 0.8493, 0.8516, 0.8388, 0.8399, 0.8362, 0.8284, 0.7837, 0.5003],
            "Recall": [0.9241, 0.9316, 0.9192, 0.9185, 0.9063, 0.9063, 0.9020, 0.9012, 0.8965, 0.9084, 0.9053, 0.9018, 0.8831, 0.7813, 1.0000]
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
        **Principais Features:**
        1. SessionsPerWeek (Importância: 98%)
        2. PlayerLevel (2%)
        3. AchievementsUnlocked (2%)
        
        **Transformações:**
        - One-Hot Encoding nas variáveis categóricas
        - Standard Scaling na variáveis numéricas
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

        st.markdown(""" Embora este seja o único gráfico apresentado, todas as variáveis categóricas analisadas seguem o mesmo padrão de distribuição, 
        mantendo uma proporcionalidade equilibrada entre suas categorias que reflete quase que diretamente a quantidade de observações em cada classe.
        """)
        
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
        
        st.markdown(""" Novamente, embora tenhamos destacado apenas este gráfico específico, todas as análises entre variáveis contínuas revelaram o mesmo padrão: 
        uma dispersão aleatória de pontos que não indica qualquer correlação significativa entre as variáveis analisadas nem com os níveis de engajamento dos 
        jogadores.
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
        
        st.markdown(""" As correlações entre as variáveis numéricas são notavelmente fracas (próximas de zero em vários casos), o que explica diretamente sua baixa 
        importância no modelo de classificação do engajamento dos jogadores.
        """)
    
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
        ### 1. Seleção de Features
        Foram removidas as seguintes variáveis:
        - **PlayerID:** Identificador único sem valor preditivo
        - **AvgSessionDurationMinutes:** Eliminada para evitar multicolinearidade com `SessionsPerWeek` e `PlayTimeHours`
        - **Gender e Location:** Removidas após análise de importância de features por piorarem decimalmente o desempenho dos modelos testados
        """)

        st.markdown("""
        ### 2. Transformação de Variáveis
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
        
        st.header("📋 Dados Pré-processados (Amostra)")
        st.dataframe(dados_prep.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros", len(dados_prep))
        with col2:
            st.metric("Variáveis", len(dados_prep.columns))

# Página 4: Modelo Preditivo
elif pagina == "🤖 Modelo Preditivo":
    st.title("🤖 Modelo Preditivo: Gradient Boosting")
    st.markdown("---")

    st.header("Metodologia")
    st.markdown("""
    - **Framework:** PyCaret
    - **Seleção de Modelos:** Comparação com base no rankeamneto com o F1-Score, seguido do melhor desempenho geral entre as métricas e desempatada pela AUC
    - **Melhor Modelo:** Gradient Boosting Classifier (tunado após seleção)
    - **Métricas:**
      - Acurácia: 87%
      - Sencibilidade: 92%
      - Precisão: 84%
      - F1-Score: 88%
      - AUC: 92%
    """)
    
    tab1, tab2 = st.tabs(["🔍 Interpretação do Modelo", "🎯 Quem é o Gradient Boosting?"])
    
    with tab1:
        st.header("🔍 Interpretação do Modelo")
        try:
            model = joblib.load('model.pkl')
            st.success("✅ Modelo carregado com sucesso!")
            
            st.markdown("""
            ### Significado das Métricas
            
            **Interpretação:**  
            
            O modelo demonstra um **bom desempenho geral** (Acurácia de 87%), com destaque para sua capacidade de:  
            
            - **Capturar casos Positivos**: Alta Sencibilidade (92%) indica que o modelo identifica efetivamente jogadores engajados (apenas 8% de falsos negativos)  
            - **Distinguir Classes**: AUC (Área sib a Curva ROC) de 92% revela excelente separação entre os jogadores de baixo engajamento dos de alto engajamento 
            - **Equilíbrio**: F1-Score (88%) mostra boa harmonia entre Precisão e Recall  
            - **Chance de Erro**: Precisão (84%) sugere que, quando o modelo prevê "alto engajamento", há 16% de chance de ser falso positivo. Algo espera dada a 
            sobreposição natural nos padrões de engajamento e das variáveis preditoras apresentarem um poder limitado de discriminação  
            """)
            
            st.markdown(""" ### Importância das Variáveis""")

            feature_importance = pd.DataFrame({
                'Feature': ['SessionsPerWeek', 'PlayerLevel', 'AchievementsUnlocked', 'PlayTimeHours','Age', 
                            'InGamePurchases_1', 'EngagementLevel', 'GameGenre_RPG', 'GameGenre_Simulation', 
                            'GameGenre_Sports', 'GameGenre_Strategy', 'GameDifficulty_Hard', 'GameDifficulty_Medium'],
                'Importance': [0.98, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            })
            
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
            st.pyplot(fig)
            
            st.markdown(""" Embora, por terem uma relevãncia tão baixa na classificação do engajamento do jogador, praticamente todas as variáveis,
            por exceção de SessionPerWeek, poderiam ter sido descartadas do modelo final, mas como sua remoção teve uma mudança quase que insignificante
            aos resultados, optou-se por deixar tais variáveis com o intuito de melhorar o desempenho da tunagem dos hiperparâmetros do modelo final.""")
    
            
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")

    with tab2:
        # Seção 3: Conhecendo o Gradient Boosting
        st.header("🎯 Quem é o Gradient Boosting?")
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
            1. **Árvores Sequenciais**:  
               Cria uma série de árvores de decisão pequenas (weak learners)
            2. **Correção de Erros**:  
               Cada nova árvore foca nos resíduos (erros) da anterior
            3. **Combinação Ponderada**:  
               Resultado final é a soma das previsões de todas as árvores
            """)
            
            # st.image("https://miro.medium.com/v2/resize:fit:1400/1*_kqsmyUwK8v1gKi0tRGsCQ.gif", 
            #          caption="Fonte: Medium - Gradient Boosting em ação")
        
        with col2:

            st.markdown("""
            ### 🏆 Critério de Seleção do Modelo
            
            O modelo foi selecionado através de uma análise comparativa das métricas de desempenho, considerando inicialmente os **5 melhores modelos com base no F1-Score**. Foi atribuído um sistema de pontuação por posição:
            
            - <b>1º lugar</b> em cada métrica: <b>+2 pontos</b>
            - <b>2º lugar</b>: <b>+1 ponto</b>
            
            <br>
            
            Após essa avaliação, os dois melhores modelos ficaram <b>empatados com 5 pontos cada</b>:
            
            <div style="margin-left: 20px;">
            
            <b>1. AdaBoost Classifier</b>  
               - 🥇 <b>Melhor desempenho</b> em:  
                 • F1-Score  
                 • Sensibilidade (Recall)  
               - 🥈 <b>Segunda melhor</b> acurácia  
            
            <b>2. Gradient Boosting</b>  
               - 🥇 <b>Melhor desempenho</b> em:  
                 • Acurácia  
               - 🥈 <b>Segundo melhor</b> em:  
                 • AUC  
                 • Sensibilidade (Recall)  
                 • F1-Score  
            </div>
            
            <h4>Critério de Desempate</h4>
            Como fator decisivo, foi considerado o <b>maior valor de AUC</b> (Area Under the Curve) do <i>Gradient Boosting</i>, uma vez que a variável resposta
            <b>não apresenta limites bem definidos entre suas categorias</b>. Nesse contexto, um modelo com maior capacidade de <b>distinguir as classes</b> 
            (refletido pelo AUC mais alto) é preferível.
            
            """, unsafe_allow_html=True)
            st.success(" **Observação Final:💡As diferenças entre as métricas dos dois modelos são muito sutis, não havendo um desempenho significativamente superior de um em relação ao outro. A escolha final priorizou a robustez na discriminação das categorias.**")
        # Detalhes técnicos com expansor
        with st.expander("🧮 A Matemática por Trás", expanded=False):
            st.markdown("""
            **Função Objetivo**:
            ```
            F(x) = γ₁h₁(x) + γ₂h₂(x) + ... + γₙhₙ(x)
            ```
            Onde:
            - `hₙ(x)`: Árvore individual (weak learner)
            - `γₙ`: Peso de cada árvore (aprendido durante o treino)
            
            **Passo a Passo**:
            1. Inicia com predição ingênua (média)
            2. Calcula resíduos (erros) para cada observação
            3. Treina nova árvore para prever esses resíduos
            4. Atualiza o modelo com taxa de aprendizado (η)
            5. Repete até convergência ou limite de iterações
            """)
        with st.expander("🔧 Configuração Técnica Detalhada", expanded=False):
            st.code("""
            GradientBoostingClassifier(
                ccp_alpha=0.0,                 # Sem poda de complexidade adicional
                criterion='friedman_mse',      # Método para encontrar melhores splits (considera valores médios)
                learning_rate=0.001,           # Taxa de aprendizado cuidadosa
                max_depth=6,                   # Profundidade controlada
                max_features='log2',           # Otimização para muitas features
                min_samples_leaf=3,            # Prevenção de overfitting
                min_impurity_decrease=0.0005,  # Explicação do mecanismo de poda automática
                n_estimators=60,               # Número ideal de árvores
                subsample=0.95,                # Stochastic Gradient Boosting
                random_state=42,               # Reprodutibilidade
                loss='log_loss'                # Para problemas de classificação
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
            
        with col2:
            achievements = st.slider("Conquistas Desbloqueadas", 0, 100, 30)
            difficulty = st.selectbox("Dificuldade do Jogo", ["Easy", "Medium", "Hard"], index=1)
            genre = st.selectbox("Gênero do Jogo", ["Acition", "RPG", "Simulation", "Sports", "Strategy"])
            purchases = st.radio("Realizou Compras no Jogo", ["Sim", "Não"], horizontal=True)
           
        if st.button("🔍 Prever Nível de Engajamento", type="primary", use_container_width=True):
            try:                
                # 1. Carrega o pipeline
                pipeline = joblib.load('model.pkl')
                
                # 2. Prepara os dados JÁ CODIFICADOS como o modelo espera
                input_data = pd.DataFrame({
                    'Age': [age],
                    'PlayTimeHours': [play_time],
                    'SessionsPerWeek': [sessions],
                    'PlayerLevel': [level],
                    'AchievementsUnlocked': [achievements],
                    
                    # Variáveis categóricas JÁ CODIFICADAS (one-hot)
                    'GameDifficulty_Hard': [1 if difficulty == "Hard" else 0],
                    'GameDifficulty_Medium': [1 if difficulty == "Medium" else 0],
                    'GameGenre_RPG': [1 if genre == "RPG" else 0],
                    'GameGenre_Simulation': [1 if genre == "Simulation" else 0],
                    'GameGenre_Sports': [1 if genre == "Sports" else 0],
                    'GameGenre_Strategy': [1 if genre == "Strategy" else 0],
                    'InGamePurchases_1': [1 if purchases == "Sim" else 0]
                })
                
                # 3. Garante a ordem correta das colunas
                input_data = input_data[pipeline.named_steps['actual_estimator'].feature_names_in_]
                st.write("Classes do modelo:", pipeline.classes_)
                st.write("Feature names:", pipeline.named_steps['actual_estimator'].feature_names_in_)
                st.write(pipeline.named_steps)
                try:
                    # 4. Faz a previsão (o imputer vai lidar com quaisquer valores faltantes)
                    prediction = pipeline.predict(input_data)[0]
                    proba = pipeline.predict_proba(input_data)[0][1]
                    
                    st.success(f"Previsão: {prediction} (Probabilidade: {proba:.2%})")
                    
                except Exception as e:
                    st.error(f"Erro na previsão: {str(e)}")
                    st.write("Dados enviados:", input_data)
                    st.write("Features esperadas:", pipeline.named_steps['actual_estimator'].feature_names_in_)
                
                # Exibir resultados
                st.markdown("---")
                st.subheader("📊 Resultado da Previsão")
                
                if prediction == 1:
                    st.success(f"## Alto Engajamento ({proba:.1%} de confiança)")
                    st.balloons()
                else:
                    st.warning(f"## Baixo Engajamento ({(1-proba):.1%} de confiança)")
                
                # Gráfico de probabilidade
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['Probabilidade'], [proba], color='#4ECDC4' if prediction == 1 else '#FF6B6B')
                ax.set_xlim(0, 1)
                ax.axvline(0.5, color='gray', linestyle='--')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Erro na previsão: {str(e)}")
                st.info("""
                Possíveis soluções:
                1. Recrie o modelo garantindo que 'EngagementLevel' seja apenas a target
                2. Verifique se todas as features estão na ordem correta
                """)
# Rodapé
st.markdown("---")
st.caption("Desenvolvido com base nas análises de pré-processamento do notebook disponíveis no [GitHub](https://github.com/thaisar13/Game-Engagement-Level)")
