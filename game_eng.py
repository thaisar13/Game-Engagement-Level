"""## **Análise Exploratória dos Dados**"""

# Instalando biblioteca de visualização
!pip install numpy==1.23.5
!pip install --upgrade sweetviz

"""### **Usando PyCaret**"""

!pip install pycaret

import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("rabieelkharoua/predict-online-gaming-behavior-dataset")
csv_path = os.path.join(path, "online_gaming_behavior_dataset.csv")
dados = pd.read_csv(csv_path)

# Filtrar apenas níveis Fácil e Difícil
dados = dados[dados['EngagementLevel'].isin(['Low', 'High'])]

dados.head()

import sweetviz as sv
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')

# Alterado target_feat para GameDifficulty
eda = sv.analyze(source=dados, target_feat='GameDifficulty')
eda.show_html()

"""## **Pré-processamento para Machine Learning**

### **PP1 - Descarte de variáveis não importantes**
"""

# Mantendo variáveis relevantes para dificuldade do jogo
dados.drop(['PlayerID', 'AvgSessionDurationMinutes', 'GameGenre', 'Location', 'InGamePurchases'], axis=1, inplace=True)
dados.head()

"""### **PP2 - Verificação de dados faltantes**"""

dados.isnull().sum()

"""### **PP3 - Dummyficação de Variáveis**"""

# Converter GameDifficulty para binário (Easy=0, Hard=1)
dados['EngagementLevel'] = dados['EngagementLevel'].map({'Low': 0, 'High': 1})

# Aplicar get_dummies nas outras variáveis categóricas
variaveis_cat = [ 'GameDifficulty', 'Gender']
dados = pd.get_dummies(dados, columns=variaveis_cat, drop_first=True)

dados.head()

"""### **PP4 - Escala das Variáveis Contínuas**"""

from sklearn.preprocessing import StandardScaler

# Padronizar variáveis numéricas
#scaler = StandardScaler()
#numericas = ['Age', 'SessionsPerWeek', 'PlayTimeHours', 'AchievementsUnlocked', 'PlayerLevel']
#dados[numericas] = scaler.fit_transform(dados[numericas])

print(dados.head())

"""## **Machine Learning**"""

from pycaret.classification import *

# Configurar ambiente com GameDifficulty como target
exp = setup(
    data=dados,
    target='EngagementLevel',
    train_size=0.75,
    session_id=42,
    normalize=True#,
    #feature_selection=True
)

# Comparar modelos
top_models = compare_models(sort='F1', n_select=3)

# Criar e ajustar o melhor modelo
best_model = create_model(top_models[0])

# Tunagem de hiperparâmetros
tuned_model = tune_model(best_model, optimize='F1')

# Avaliar modelo
evaluate_model(tuned_model)

# Finalizar e salvar modelo
final_model = finalize_model(tuned_model)
save_model(final_model, 'modelo')
