# 🌟 Seminário – **XGBoost**

Experimento Prático no *Wisconsin Breast Cancer Dataset*

---

## 📑 Sumário

1. [Motivação](#motivação)  
2. [Fundamentos Teóricos](#fundamentos-teóricos)  
3. [Dataset & Estratificação](#dataset--estratificação)
4. [Análise Exploratória (EDA)](#análise-exploratória-eda)  
5. [Pré-processamento](#pré-processamento)  
6. [Particionamento Estratificado](#particionamento-estratificado)  
7. [Baseline & Tuning](#baseline--tuning)
8. [Avaliação](#avaliação)  
9. [Interpretabilidade](#interpretabilidade)  
10. [Reflexão Crítica](#reflexão-crítica)  
11. [Reproduzindo o Estudo](#reproduzindo-o-estudo)
12. [Artefatos](#artefatos)
13. [Referências](#referências)

---

## Motivação

> **Objetivo principal:** mostrar, na prática, todo o *pipeline* com **XGBoost**, compará-lo ao SVM (seminário anterior) e discutir resultados, interpretabilidade e trade-offs de uso em produção.

---

## Fundamentos Teóricos

O **XGBoost** (eXtreme Gradient Boosting) implementa *gradient boosting* com:

- **Regularização L1/L2** para conter *overfitting*.
- **Paralelização** por bloco + suporte a GPU.
- ***Early Stopping*** nativo.

### Função objetivo

```math

\mathcal{L}(\theta)=\sum_{i=1}^{n} l\!\bigl(y_i,\hat{y}_i\bigr)+\sum_{k=1}^{K}\Omega(f_k), \\ Onde: \quad \Omega(f)=\gamma T + \tfrac{\lambda}{2}\lVert w\rVert^{2}
```

| Parâmetro            | Papel                                              | Pontos de partida |
|----------------------|----------------------------------------------------|-------------------|
| `n_estimators`       | Nº de árvores                                      | 200 – 400 + `early_stopping_rounds` |
| `learning_rate`      | Contribuição por árvore                            | 0.01 – 0.1 |
| `max_depth`          | Profundidade máxima                                | 3 – 6 |
| `subsample`          | % linhas por árvore                                | 0.7 – 1.0 |
| `colsample_bytree`   | % colunas por árvore                               | 0.7 – 1.0 |
| `gamma`              | Ganho mínimo p/ split                              | 0 – 5 |
| `lambda`, `alpha`    | Regularização L2 / L1                              | 0 – 10 |

---

## Dataset & Estratificação

- **Fonte:** *Breast Cancer Wisconsin* (`sklearn.datasets.load_breast_cancer`)  
- **Classes:** 0 = Maligno · 1 = Benigno  
- **Estratificação** garantiu a mesma proporção de classes em treino e teste:

| Split | Amostras | % Benigno |
|-------|----------|-----------|
| Treino| 910      | 63 % |
| Teste | 114      | 63 % |

---

## Análise Exploratória (EDA)

```python

import seaborn as sns, matplotlib.pyplot as plt
sns.countplot(x='target', data=df); plt.title('Distribuição das Classes')
```

- Correlações fortes entre `mean_*` e `worst_*`.  
- Nenhum valor ausente.  
- Algumas features altamente colineares → regularização lida bem, mas vale monitorar.

---

## Pré-processamento

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

> *Embora árvores não exijam escala, mantivemos para comparabilidade com o SVM.*

---

## Particionamento Estratificado

```python
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.111, random_state=42)
```

- Proporções conservadas (ver tabela acima).  
- `random_state` fixado → reprodutibilidade.

---

## Baseline & Tuning

### Baseline rápido

```python
from xgboost import XGBClassifier
clf0 = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric='logloss', use_label_encoder=False)
clf0.fit(X_train, y_train)
```

### Tuning com `RandomizedSearchCV`

```python
param_grid = { ... }  # ver notebook
cv = StratifiedKFold(10, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    param_grid, n_iter=50, cv=cv, scoring='roc_auc', random_state=42,
    n_jobs=-1, verbose=0)
search.fit(X_train, y_train,
           eval_set=[(X_test, y_test)],
           early_stopping_rounds=30, verbose=False)
best_clf = search.best_estimator_
```

---

## Avaliação

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)

y_pred = best_clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
RocCurveDisplay.from_estimator(best_clf, X_test, y_test)
PrecisionRecallDisplay.from_estimator(best_clf, X_test, y_test)
```

| Métrica  | Valor |
|----------|-------|
| AUC-ROC  | **0.99** |
| Precisão | 0.96 |
| Recall   | 0.95 |
| F1-score | 0.95 |

Confusion Matrix: 2 FP · 3 FN.

---

## Interpretabilidade

### Importância de Features nativa

```python
from xgboost import plot_importance
plot_importance(best_clf, max_num_features=10)
```

### SHAP

```python
import shap
explainer = shap.TreeExplainer(best_clf)
shap_vals = explainer.shap_values(X_test)
shap.summary_plot(shap_vals, X_test, plot_type="dot")
```

- `worst_area`, `worst_radius` e `worst_concave_points` dominam a contribuição.  
- Valores altos dessas variáveis aumentam probabilidade de maligno.

---

## Reflexão Crítica

| Questão                           | Insight |
|----------------------------------|---------|
| **Overfitting**                  | Não foi observado: *early stopping* + regularização mantiveram gap < 1 pp entre treino e validação. |
| **Comparação com SVM (seminário)** | XGBoost ↓ AUC (-0.0264), ↓ tempo de predição (~4×). |
| **Limitações**                   | Tuning extenso; riscos de *data leakage* se *pipeline* não for bem fechado. |
| **Próximos Passos**              | LightGBM / CatBoost; *stacking* com SVM; calibração de probabilidades (`CalibratedClassifierCV`). |

---

## Reproduzindo o Estudo

```bash
# 1 · Clone o repositório
git clone https://github.com/thalesfb/machine_learning.git
cd machine_learning/seminar/xgboost

# 2 · Ambiente
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3 · Execute
jupyter lab xgboost.ipynb
```

> **Versão Python ≥ 3.10** recomendada.

---

## Artefatos

| Arquivo | Propósito |
|---------|-----------|
| [**`xgboost.ipynb`**](./xgboost.ipynb) | Notebook completo (código + gráficos). |
| [**`slides`**](https://docs.google.com/presentation/d/1CKbIO8EjqNqdgZhZIB3_YTu7wHhXNLksIFju-FgbMLk/edit#slide=id.p) | Apresentação de 15 min. |
| [**`requirements.txt`**](./requirements.txt) | Lista de dependências (≈ 80 MB instalação clean). |
| **Dataset** | Carregado diretamente via *scikit-learn* (sem download manual). |

---

## Referências

- 🔗[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- 🔗[Breast Cancer Wisconsin Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 🔗[XGBoost Docs](https://xgboost.readthedocs.io/)
- 🔗[StatQuest: Gradient Boosting Explained](https://youtu.be/3CC4N4z3GJc)
- 🔗[Kaggle XGBoost Tutorials](https://www.kaggle.com/tag/xgboost)
- 🔗[Scikit-learn Docs](https://scikit-learn.org/stable/index.html)
- 🔗[Bias-Variance Tradeoff](https://scott.fortmann-roe.com/docs/BiasVariance.html)

---

> *“All models are wrong, but some are useful.” – George Box*  
> **Faça a validação cruzada sempre.**
