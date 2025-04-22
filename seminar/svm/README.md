# Seminário: Aplicação do Algoritmo SVM com Análise de Resultados

---

## Introdução

Este projeto explora o uso do algoritmo Support Vector Machine (SVM) para classificação, com foco em aplicações práticas, interpretação dos resultados e dicas para experimentação. O SVM é amplamente utilizado em machine learning devido à sua robustez e capacidade de lidar com dados de alta dimensão.

---

## 1. Fundamentos Teóricos

### O que é SVM?

O Support Vector Machine (SVM) é um algoritmo de aprendizado supervisionado que busca encontrar o melhor hiperplano para separar classes distintas em um conjunto de dados. O objetivo é maximizar a margem, ou seja, a distância entre o hiperplano e os pontos mais próximos de cada classe (os chamados vetores de suporte).

- **Margem**: Quanto maior a margem, maior a confiança na separação das classes.
- **Hiperplano**: Fronteira de decisão que separa as classes.
- **Vetores de suporte**: Pontos que estão mais próximos do hiperplano e determinam sua posição.

Se os dados não forem linearmente separáveis, o SVM utiliza funções kernel para projetar os dados em um espaço de maior dimensão, onde a separação é possível.

### Kernels

- **Linear**: Para dados linearmente separáveis.
- **Polinomial**: Útil para problemas não lineares com características polinomiais.
- **RBF (Radial Basis Function)**: Muito utilizado, cria fronteiras não lineares e é versátil para diferentes tipos de dados.
- **Sigmoid**: Menos comum, similar ao usado em redes neurais.

A escolha do kernel pode impactar fortemente o desempenho do modelo. O kernel RBF é geralmente um bom ponto de partida.

### Parâmetros Importantes

- **C**: Controla o trade-off entre maximizar a margem e minimizar o erro de classificação. Valores altos de C buscam classificar todos os pontos corretamente (menos margem), enquanto valores baixos permitem mais erros, mas com margem maior.
- **gamma**: Define o alcance de influência de um único ponto de treinamento. Valores altos de gamma fazem com que apenas pontos próximos influenciem a decisão, enquanto valores baixos consideram pontos mais distantes.

Ajustar esses parâmetros é fundamental para obter um bom desempenho.

### Vantagens e Desvantagens

- **Vantagens**: Eficaz com dados de alta dimensão, robustez contra overfitting, flexível devido à escolha de diferentes kernels.
- **Desvantagens**: Sensível a parâmetros e a datasets grandes ou muito desbalanceados, pode ser lento em grandes volumes de dados.

### Aplicações Práticas

- Segurança (reconhecimento facial)
- Saúde (diagnóstico médico)
- Reconhecimento de padrões
- Análise de sentimentos

---

## Fundamentação Matemática

O SVM busca encontrar o hiperplano que melhor separa as classes, maximizando a margem entre os pontos mais próximos de cada classe (vetores de suporte). Os principais conceitos matemáticos são:

- **Hiperplano**: É a fronteira de decisão que separa as classes. Em um espaço de $d$ dimensões, é definido por $\mathbf{w}^T \mathbf{x} + b = 0$, onde $\mathbf{w}$ é o vetor normal e $b$ o viés.
- **Margem**: Distância entre o hiperplano e os vetores de suporte. O SVM maximiza essa margem para garantir maior generalização.
- **Formulação Primal**: O problema é formulado como uma minimização da norma de $\mathbf{w}$, sujeita a restrições que garantem a separação correta das classes.
- **Formulação Dual**: Utiliza multiplicadores de Lagrange para transformar o problema, facilitando o uso de funções kernel e permitindo trabalhar em espaços de alta dimensão.
- **Kernel Trick**: Permite que o SVM encontre fronteiras de decisão não lineares ao projetar os dados em um espaço de maior dimensão, sem calcular explicitamente essa projeção. Exemplos: linear, polinomial, RBF, sigmoid.
- **Soft Margin**: Introduz variáveis de folga e um parâmetro de penalização $C$ para lidar com dados não perfeitamente separáveis, equilibrando margem e erros de classificação.

Esses conceitos garantem que o SVM seja robusto, flexível e eficaz em diferentes cenários de classificação.

---

## 2. Experimento Prático

### Dataset Escolhido: [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

#### Descrição

- **Problema**: Classificação entre tumores benignos e malignos.
- **Variáveis**: 30 características numéricas extraídas de imagens digitalizadas.

#### Por que este dataset?

É um benchmark clássico em machine learning, com classes bem definidas e características numéricas, ideal para explorar o funcionamento do SVM.

### Passo a Passo do Experimento

1. **Carregamento e análise exploratória dos dados**
2. **Pré-processamento**: tratamento de dados faltantes e normalização com StandardScaler
3. **Divisão do dataset**: treino (80%) e teste (20%), validação cruzada k-fold
4. **Treinamento do modelo**: Ajuste de hiperparâmetros com GridSearchCV (C, gamma e kernel)
5. **Avaliação**: acurácia, matriz de confusão, precisão, recall, F1-score, AUC-ROC
6. **Discussão dos resultados**: análise crítica e sugestões de melhorias

### Como Executar o Notebook

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Abra o notebook `svm.ipynb` em seu ambiente Jupyter ou VS Code.
3. Execute as células sequencialmente.

### Dicas Práticas

- Sempre normalize os dados antes de treinar o SVM.
- Use validação cruzada para avaliar o modelo e evitar overfitting.
- Ajuste os hiperparâmetros (C, gamma, kernel) com GridSearchCV.
- Analise a matriz de confusão para entender os erros do modelo.

### Sugestões de Exploração Adicional

- Teste outros kernels (linear, polinomial) e compare os resultados.
- Aplique técnicas de balanceamento de classes (undersampling, oversampling).
- Explore a importância das features usando métodos de seleção de variáveis.
- Analise o impacto de diferentes estratégias de normalização.

## Artefatos e Resultados

- [**Apresentação SVM**](https://docs.google.com/presentation/d/1KPVmIJ24JEXdTuvchBfo1M4ny5204INyuXNw5aXr4Pg/edit#slide=id.p)
- [**Notebook**](https://github.com/thalesfb/machine_learning/blob/main/seminar/svm/svm.ipynb)

## Referências

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [DataCamp: Tutorial de Vetor de Suporte com o Scikit-learn](https://www.datacamp.com/pt/tutorial/svm-classification-scikit-learn-python)
- [IBM: Tutorial Prático sobre SVM com Kernel RBF](https://www.ibm.com/think/topics/support-vector-machine)
- [Support Vector Machines: Theory and Applications](https://www.researchgate.net/publication/221621494_Support_Vector_Machines_Theory_and_Applications)
