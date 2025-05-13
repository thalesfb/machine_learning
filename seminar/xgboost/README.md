# 🌟 Seminário – **XGBoost** do Zero ao Avançado  
<sub>Experimento prático no Wisconsin Breast Cancer Dataset</sub>

---

## 📘 Recomendações Rápidas

| ✔️ Como usar | 💡 Dicas extras |
|-------------|----------------|
| Execute as células do notebook na ordem. | Altere hiperparâmetros e observe o impacto nas métricas. |
| Mantenha um conjunto de validação (`early_stopping_rounds`). | Compare com SVM ou Random Forest para sentir diferenças. |
| Explore `scale_pos_weight` se as classes estiverem desbalanceadas. | Use SHAP para explicar predições individuais. |

---

## 🎯 Objetivos

1. **Entender, na teoria, como o XGBoost funciona** (boosting, regularização, paralelização).  
2. **Implementar um pipeline completo**: EDA → Pré-processamento → Treino → Avaliação → Interpretação.  
3. **Comparar desempenho** com o SVM do seminário anterior e discutir vantagens e limitações.  

> ✨ *Objetivo principal:* Entender na prática o funcionamento do XGBoost!

---

## 🔍 1. Fundamentos Teóricos

O **XGBoost** (“eXtreme Gradient Boosting”) é uma técnica poderosa baseada em boosting de gradiente, muito utilizada em competições de Machine Learning e na indústria devido ao seu alto desempenho e eficiência.

> 🎉 É uma evolução de algoritmos de boosting tradicionais, trazendo regularização e paralelização para o processo.

### 1.1 O que é XGBoost?

- **Boosting** = combinação sequencial de modelos fracos (árvores rasas).  
- **Gradient boosting** = cada árvore minimiza o gradiente da perda acumulada.  
- **XGBoost** = implementação otimizada com regularização \(L_1/L_2\) e paralelização por blocos.

> 🔹 **Curiosidade:** O XGBoost é considerado "a arma secreta" em competições do Kaggle!

### 🧮 1.2 Matemática Essencial

$$
\begin{aligned}
\mathrm{Obj}(\Theta)
&= \sum_{i=1}^{n} l\bigl(y_i,\hat{y}_i\bigr)
    + \sum_{k=1}^{K}\Omega(f_k),\\
Onde: \\
\Omega(f)
&= \gamma\,T \;+\; \tfrac{1}{2}\,\lambda\,\|w\|^{2}
\end{aligned}
$$

Onde:

- Θ = conjunto de parâmetros do modelo
- \(n\) = nº de amostras
- \(l\) = função de perda (log-loss neste dataset).
- γ = requisito mínimo de ganho por split
- ŷᵢ = predição
- \(K\) = nº de árvores.
- Ω = função de complexidade (termo de regularização) aplicado a cada árvore  
- \(f_k\) = árvore \(k\)
- \(T\) = nº de folhas
- \(w\) = pesos das folhas
- λ = regularização L2
- α = regularização L1

### 1.3 Parâmetros-chave

| Parâmetro | Efeito | Recomendações iniciais |
|-----------|--------|------------------------|
| `n_estimators` | Nº de árvores | 100–300 (+ `early_stopping_rounds`) |
| `learning_rate` | Peso de cada árvore | 0.01–0.2 (quanto menor, mais árvores) |
| `max_depth` | Profundidade | 3–7 (maior → +complexo) |
| `subsample` | % de linhas por árvore | 0.6–1.0 (previne overfitting) |
| `colsample_bytree` | % de colunas por árvore | 0.6–1.0 |
| `gamma` | Mín. ganho p/ split | 0–5 |
| `lambda`, `alpha` | Reg. L2 e L1 | 0–10 |

### 1.4 Vantagens × Desvantagens

| 💪 Vantagens | ⚠️ Desvantagens |
|--------------|----------------|
| State-of-the-art em dados tabulares; lida com `NaN`. | Muitos hiperparâmetros; tuning pode ser demorado. |
| Treino rápido (CPU/GPU) e custo log-loss menor. | Pode sobreajustar se pouco regularizado. |
| Integra regularização e *early stopping*. | Menos interpretável que modelos lineares. |

### 1.5 Casos de Uso Reais

Finanças (risco de crédito), saúde (diagnóstico assistido), marketing (churn), detecção de fraude, previsão de demanda.

---

## 📊 2. Análise Exploratória de Dados (EDA)

```python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# Carregar dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
df['target'] = data.target

# Visualizar primeiras linhas
df.head()
```

```python
# Distribuição das classes
sns.countplot(x='target', data=df)
plt.title('Distribuição das Classes')
plt.show()

# Correlação entre variáveis
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.show()

df.info()

# Verificar dados faltantes
df.isnull().sum()

# Verificar dados duplicados
df.duplicated().sum()

# Verificar estatísticas descritivas
df.describe()
```

## 🛠️ 3. Pré-processamento

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Remover duplicados
df_clean = df.drop_duplicates()
df_clean.reset_index(drop=True, inplace=True)

# Separar features e target
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

```

>🗒️ Por serem árvores, a padronização não é obrigatória; mantivemos para alinhar com o SVM anterior.

## 🚀 4. Treinamento e Tuning

```python
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
params = {
    'max_depth': [3,5,7],
    'n_estimators': [100,200,300],
    'learning_rate': [0.01,0.05,0.1],
    'subsample': [0.8,1.0],
    'colsample_bytree': [0.8,1.0],
    'gamma': [0,1,5],90
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    params, n_iter=20, cv=cv, scoring='roc_auc', random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          early_stopping_rounds=20, verbose=False)
best_clf = model.best_estimator_

```

> 🔹 **Dica:** Teste diferentes hiperparâmetros para melhorar a performance!

## 📈 5. Avaliação

```python
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

# Previsão
y_pred = model.predict(X_test)

# Métricas
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# AUC-ROC
roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()

```

### 5.1 Importância das Features

```python
import xgboost as xgb, shap
xgb.plot_importance(best_clf, max_num_features=10)
explainer = shap.TreeExplainer(best_clf)
shap.summary_plot(explainer.shap_values(X_test), X_test, plot_type="bar")

```

## 🧐 6. Análise Crítica

| Questão                | Observação                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------- |
| **Overfitting**        | Foi mitigado por `early_stopping_rounds` + regularização?                               |
| **Comparação com SVM** | XGBoost superou AUC-ROC? Vale o custo de complexidade?                                  |
| **Melhorias Futuras**  | Testar *ensemble* stacking SVM + XGBoost; ajustar `scale_pos_weight`; avaliar LightGBM. |

> 🔹 **Reflexão:** O XGBoost foi eficaz para este problema? Por quê?

## ⚙️ Como Reproduzir

```bash
# 1. Ambiente
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Notebook
jupyter lab  # ou VSCode / Google Colab

```

## 🚀 Artefatos

| Arquivo              | Descrição                                   |
| -------------------- | ------------------------------------------- |
| `xgboost.ipynb`      | Notebook completo com códigos e gráficos    |
| `slides_xgboost.pdf` | Slides de 15 min usados na apresentação     |

## 📚 Referências

- 🔗[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- 🔗[Breast Cancer Wisconsin Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 🔗[XGBoost Docs](https://xgboost.readthedocs.io/)
- 🔗[StatQuest: Gradient Boosting Explained](https://youtu.be/3CC4N4z3GJc)
- 🔗[Kaggle XGBoost Tutorials](https://www.kaggle.com/tag/xgboost)
- 🔗[Scikit-learn Docs](https://scikit-learn.org/stable/index.html)
- 🔗[Bias-Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html)