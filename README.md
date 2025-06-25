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

Após extensiva avaliação comparativa entre diversos algoritmos de Machine Learning, o **Ada Boost Classifier** demonstrou o melhor desempenho, sendo em seguida otimizado resultando nos seguintes valores de parâmetros e métricas:

```{python}
AdaBoostClassifier(
                algorithm='SAMME.R',       # Versão real do algoritmo AdaBoost
                base_estimator=None,        # Por padrão usa DecisionTree com max_depth=1 (stump)
                learning_rate=1.0,          # Taxa de aprendizado (contribuição de cada modelo)
                n_estimators=50,           # Número de stumps (modelos fracos)
                random_state=42             # Reprodutibilidade
            )
```
| Métrica       | Valor | Interpretação                                                                 |
|--------------|-------|-------------------------------------------------------------------------------|
| **F1-Score** | 0.88  | **"Harmonia" entre precisão e recall** - Pontuação balanceada considerando ambos os aspectos |
| **Acurácia** | 0.87  | **Acertos totais** - Percentual geral de classificações corretas              |
| **Precisão** | 0.83  | **Confiança nos positivos** - Quando prevê "Alto Engajamento", acerta 83% das vezes |
| **Sencibilidade**   | 0.93  | **Cobertura de positivos** - Detecta 93% dos casos reais de Alto Engajamento  |


## 💡 Conhecendo o Gradient Boosting Classifier

O **Ada Boost Classifier** foi selecionado como algoritmo principal por oferecer o melhor balanceamento entre desempenho preditivo e robustez estatística. Suas características técnicas se alinham perfeitamente com nosso cenário de dados:

### 🎯 Mecanismo de Funcionamento do Ada Boost

1. **Passo Inicial - Base Simples**
   
  - Começa com um modelo fraco (ex: árvore de decisão rasa - stump)
  - Todos os exemplos têm peso igual inicialmente

2. **Processo Iterativo - Aprendizado com Erros**
   
   *Primeira Iteração*:
     - O stump faz predições iniciais
     - Erros são identificados e os exemplos mal classificados recebem mais peso

   *Iterações Seguintes*:
     - Cada novo stump foca nos exemplos mais difíceis (com maior peso)
     - Modelos subsequentes "herdam" os erros corrigidos anteriormente

4. **Mecanismo de Peso**
   
  - **Peso dos Exemplos**: Aumenta para casos mal classificados
  - **Peso dos Modelos**: Stumps mais precisos têm maior influência no voto final

4. **Resultado Final - Voto Ponderado**
   
  - Combina todas as previsões dos stumps
  - Cada contribuição é ponderada pela precisão do modelo

5. **Vantagens Chave**
   
  - Foco automático nos casos mais difíceis
  - Simples e eficaz para problemas binários
  - Menos propenso a overfitting que algoritmos complexos
            
## 🏆 Comparative Model Performance Analysis

| Rank | Model                          | Acurácia    | AUC       | Sencibilidade    | Precisão | F1-Score   | Tempo de Treinamento|
|------|--------------------------------|-------------|-----------|------------------|-----------|------------|---------------|
| 🥇   | **Ada Boost Classifier**        |    **0.8719**	  |  *0.9156*	  |  	**0.9316**		     |  0.8324		  |  **0.8792**	  |  	0.6960
| 🥈   | *Gradient Boosting Classifier*	|    *0.8711*	  |  **0.9164**	  |  	*0.9230*	       |  	*0.8365*	  |  	*0.8776*	  |  	1.8600
| 🥉   | Light Gradient Boosting Machine|	  0.8707	  |  0.9134	  |  	0.9192	   	  |  0.8381		  |  **0.8767**	  |  	2.0800


### Legenda:
- **Negrito** = Melhor resultado na métrica
- *Itálico* = Segundo melhor resultado

Outro fator que influenciou na escolha do AdaBoost Classifier foi a distribuição de importância das variáveis. No Gradient Boosting Classifier, a variável 'SessionsPerWeek' apresentava 97% de importância, reduzindo 'PlayTimeHour' a uma relevância quase nula - um padrão inadequado, pois: um jogador pouco engajado pode ter várias sessões semanais mas com poucas horas jogadas em cada, enquanto um jogador altamente engajado pode acumular muitas horas de jogo em poucas sessões prolongadas. O AdaBoost, ao distribuir melhor essa importância, captura essa nuance comportamental de forma mais equilibrada.

## 🚀 Experimente Agora!
Acesse a versão interativa do modelo:  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nivelengajamentojogo.streamlit.app).
