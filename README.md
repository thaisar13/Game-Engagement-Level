# 🎮 Game Engagement Prediction App

## 📦 Dependencias Necessarias

```python
pip install -r requirements.txt
```

## 🎯 Objetivo do Projeto
Prever com alta precisão o nível de engajamento (Alto/Baixo) que um jogo terá com os jogadores, analisando:

- Comportamento do jogador (horas jogadas, frequência semanal)
- Características do jogo (gênero, dificuldade)
- Progresso do jogador (nível atingido, nº de conquistas)

## 🔍 Seleção do Modelo

Após extensiva avaliação comparativa entre diversos algoritmos de Machine Learning, o **Gradient Boosting Classifier** demonstrou o melhor desempenho, sendo em seguida otimizado resultando nos seguintes valores de parâmetros e métricas:

```{python}
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.001, loss='log_loss', max_depth=6,
                           max_features='log2', max_leaf_nodes=None,
                           min_impurity_decrease=0.0005, min_samples_leaf=3,
                           min_samples_split=4, min_weight_fraction_leaf=0.0,
                           n_estimators=60, n_iter_no_change=None,
                           random_state=42, subsample=0.95, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
```
| Métrica       | Valor | Interpretação                                                                 |
|--------------|-------|-------------------------------------------------------------------------------|
| **F1-Score** | 0.88  | **"Harmonia" entre precisão e recall** - Pontuação balanceada considerando ambos os aspectos |
| **Acurácia** | 0.87  | **Acertos totais** - Percentual geral de classificações corretas              |
| **Precisão** | 0.84  | **Confiança nos positivos** - Quando prevê "Alto Engajamento", acerta 83% das vezes |
| **Sencibilidade**   | 0.92  | **Cobertura de positivos** - Detecta 93% dos casos reais de Alto Engajamento  |


## 💡 Conhecendo o Gradient Boosting Classifier

O **Gradient Boosting Classifier** foi selecionado como algoritmo principal por oferecer o melhor balanceamento entre desempenho preditivo e robustez estatística. Suas características técnicas se alinham perfeitamente com nosso cenário de dados:

### 🎯 Mecanismo de Funcionamento do Gradient Boosting
1. **Otimização por Gradiente**:

O modelo funciona através de um processo iterativo que:

   - Começa com uma previsão inicial simples (ex: média)
   - Calcula os erros (gradientes) entre previsões e valores reais
   - Treina uma nova árvore para prever esses erros
   - Atualiza as previsões com pequenos passos controlados (learning_rate=0.001)
   - Repete por 60 iterações (n_estimators=60)

Fórmula Chave:
Nova Previsão = Previsão Anterior + learning_rate * Previsão da Árvore-Atual

2. **Aprendizado Sequencial em Camadas**:

Passo-a-passo:
   1) Primeira árvore:
      - Faz previsão inicial (baseline)
      - Calcula erros (valores reais - previsões)

   2) Árvores seguintes:
      - Cada nova árvore é treinada nos erros residuais das anteriores
      - Erros grandes recebem mais atenção
      - Ajustes são feitos gradualmente (passos pequenos de 0.001)

   3) Combinação final:
      - Soma ponderada de todas as 60 árvores
      - Resultado final balanceado

3. **Regularização Nativa (Anti-Overfiting)**:

Técnicas implementadas:

   a) Limitação de Profundidade (max_depth=6)
      - Árvores não podem ficar muito complexas
      - Previne memorização dos dados

   b) Subamostragem (subsample=0.95)
      - Cada árvore usa apenas 95% dos dados aleatórios
      - Aumenta diversidade das árvores

   c) Seleção de Features (max_features='log2')
      - Considera apenas √n features em cada divisão
      - Reduz correlação entre árvores


## 🏆 Comparative Model Performance Analysis

| Rank | Model                     | Acurácia    | AUC       | Sencibilidade    | Precisão | F1-Score   | Tempo de Treinamento|
|------|---------------------------|-------------|-----------|-----------|-----------|------------|---------------|
| 🥇   | **Gradient Boosting**     | **0.8720**  | *0.9167* | *0.9241*   | 0.8371    | *0.8784*  | 1.6050s      |
| 🥈   | *Ada Boost Classifier*    | *0.8718*    | 0.9156   | **0.9316** | 0.8323    | **0.8792**| 0.4470s      |
| 🥉   | LightGBM                  | 0.8712     | 0.9128    | 0.9192     | *0.8390*  | 0.8772    | 1.5580s      |
| 4️⃣   | Random Forest             | 0.8705     | 0.9082    | 0.9185     | 0.8382    | 0.8765    | 1.8570s      |
| 5️⃣   | Ridge Classifier          | 0.8716     | **0.9169**| 0.9063     | **0.8477**| 0.8760    | **0.0410s**      |

### Legenda:
- **Negrito** = Melhor resultado na métrica
- *Itálico* = Segundo melhor resultado

Note que embora o Ada Boost Classifier tenha apresentado o melhor resultado para a métrica F1-Score, que é uma métrica que balanceia os resultados de acerto do modelo classificador, o Gradient Boosting apresentou um resultado sutilmente melhor em AUC, o que indica que esse modelo apresenta uma maior capacidade de distinção das classe, e como a variável resposta não paresenta limites muito bem definidos, o Gradient Boosting foi o modelo escolhido.

## 🚀 Experimente Agora!
Acesse a versão interativa do modelo:  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nivelengajamentojogo.streamlit.app).
