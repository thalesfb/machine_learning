# Machine Learning Repository

Um repositório dedicado ao aprendizado e experimentação com algoritmos de Machine Learning e Deep Learning.

## 📋 Conteúdo do Repositório

Este repositório contém notebooks e scripts relacionados a diferentes técnicas e aplicações de machine learning:

### 🖼️ Redes Neurais Convolucionais (CNN)

- [`stl-10_classification.ipynb`](./cnn/stl-10_classification.ipynb): Implementação de classificação de imagens usando o dataset STL-10 com CNN LeNet-5.
  - Modelos treinados: [`best_model.h5`](./cnn/best_model.h5), [`efficientnet_fine_tuned.h5`](./cnn/efficientnet_fine_tuned.h5), [`efficientnet_initial.h5`](./cnn/efficientnet_initial.h5)
  - Referência: [`Lecun98.pdf`](./cnn/Lecun98.pdf) (Artigo original LeNet-5)

### 🧠 Redes Neurais Artificiais (ANN) e Modelos Clássicos

- [`cancer.ipynb`](./ann/cancer.ipynb): Aplicação de redes neurais para classificação de câncer.
- [`diabetes.ipynb`](./ann/diabetes.ipynb): Implementação de redes neurais para previsão de diabetes.
- [`heart_disease.ipynb`](./ann/heart_disease.ipynb): Implementação de redes neurais para previsão de doenças cardíacas (dataset: [`ann/heart.csv`](./ann/heart.csv)).
- [`AM_Doença_no_Coração_Naive_Bayes.ipynb`](./ann/AM_Doença_no_Coração_Naive_Bayes.ipynb): Classificação de doenças cardíacas utilizando Naive Bayes (dataset: [`ann/heart.csv`](./ann/heart.csv)).
- [`AM_Naive_Bayes_Experimento_Prático.ipynb`](./ann/AM_Naive_Bayes_Experimento_Prático.ipynb): Experimentos práticos com o algoritmo Naive Bayes.

### 📚 Material Educacional e Atividades

- [`introduction.ipynb`](./introduction.ipynb): Notebook com exercícios de revisão de conceitos de machine learning.
- [`AM_Atividade_Avaliativa_1.ipynb`](./AM_Atividade_Avaliativa_1.ipynb): Atividade avaliativa sobre Machine Learning.

### 🎓 Seminários

- **SVM (Support Vector Machines)**

  - [`svm.ipynb`](./seminar/svm/svm.ipynb): Notebook com a implementação e estudo de SVM.
  - [`README.md`](./seminar/svm/README.md): Detalhes específicos do seminário de SVM.

- **XGBoost**

  - [`xgboost.ipynb`](./seminar/xgboost/xgboost.ipynb): Notebook principal com a implementação e estudo de XGBoost.
  - [`XGBoost_exemplo.ipynb`](./seminar/xgboost/XGBoost_exemplo.ipynb): Exemplo prático de uso do XGBoost.
  - [`README.md`](./seminar/xgboost/README.md): Detalhes específicos do seminário de XGBoost.

### 🧪 Dados de Teste

- O diretório `test/` contém scripts e dados para testes, incluindo:
  - Scripts de extração de dados (`extract.py`)
  - Logs de processamento
  - Arquivos de saída em Excel
  - Scripts de visualização de dados (`visualization.py`)
  - Dados brutos em `test/data/`

## 🛠️ Bibliotecas Utilizadas

Este projeto utiliza as seguintes tecnologias:

- **Ciência de Dados**: pandas, numpy, scikit-learn, xgboost
- **Visualização**: matplotlib, seaborn, plotly
- **Deep Learning**: keras, tensorflow
- **Processamento de Imagens**: OpenCV

## 🚀 Como Usar

1. Clone o repositório
2. Instale as dependências:

   ```bash

   pip install -r requirements.txt

   ```

3. Execute os notebooks Jupyter ou scripts Python conforme necessário

## 📊 Descrição dos Projetos Principais

### Previsão de Doença Cardíaca

Modelo para prever a probabilidade de pacientes terem doenças cardíacas com base em diversos indicadores médicos como idade, sexo, tipo de dor no peito, pressão arterial, colesterol, etc. O pipeline inclui:

- Análise exploratória de dados
- Tratamento de valores ausentes e outliers
- Transformação de características
- Treinamento de modelos de classificação

### Previsão de Diabetes

Modelo para prever a probabilidade de pacientes desenvolverem diabetes com base em dados médicos. O pipeline inclui:

- Análise exploratória de dados
- Tratamento de valores ausentes e outliers
- Transformação de características
- Treinamento de modelos de classificação
- Avaliação de desempenho do modelo

### Classificação de Imagens com CNN

Implementação de redes neurais convolucionais para classificação de imagens utilizando o dataset STL-10, que contém imagens de 10 classes diferentes. A implementação utiliza a arquitetura LeNet-5 adaptada.

### Classificação de Câncer

Aplicação de redes neurais para classificação de dados de câncer, com foco em experimentos para otimização de hiperparâmetros, avaliação de diferentes arquiteturas e análise de desempenho em conjuntos de dados variados.

## 📝 Licença

Este projeto está licenciado sob os termos da licença incluída no arquivo [LICENSE](./LICENSE).
