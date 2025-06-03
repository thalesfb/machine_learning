# Machine Learning Repository

Um repositório dedicado ao aprendizado e experimentação com algoritmos de Machine Learning e Deep Learning.

## 📋 Conteúdo do Repositório

Este repositório contém notebooks e scripts relacionados a diferentes técnicas e aplicações de machine learning

### Estrutura do Repositório

```plaintext
├── ann
│   ├── cancer.ipynb
│   ├── diabetes.ipynb
│   ├── heart_disease.ipynb
│   ├── heart.csv
│   ├── AM_Doença_no_Coração_Naive_Bayes.ipynb
│   └── AM_Naive_Bayes_Experimento_Prático.ipynb
├── cnn
│   ├── stl-10_classification.ipynb
│   ├── best_model.h5
│   ├── efficientnet_fine_tuned.h5
│   ├── efficientnet_initial.h5
│   └── Lecun98.pdf
├── rnn
│   ├── models/
│   │   ├── lstm_improved_model.h5
│   │   ├── lstm_model.h5
│   ├── lstm_temperatures.ipynb
│   ├── README.md
│   ├── requirements.txt
├── seminar
│   ├── svm
│   │   ├── svm.ipynb
│   │   └── README.md
│   │   └── requirements.txt
│   ├── xgboost
│   │   ├── xgboost.ipynb
│   │   └── requirements.txt
│   │   ├── XGBoost_exemplo.ipynb
│   │   └── README.md
├── test
│   ├── extract.py
│   ├── data/
│   ├── processamento.log
│   ├── saida.xlsx
├── introduction.ipynb
├── AM_Atividade_Avaliativa_1.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

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

### 🔄 Redes Neurais Recorrentes (RNN)

- [`lstm_temperatures.ipynb`](./rnn/lstm_temperatures.ipynb): Predição de temperaturas mínimas em Melbourne usando LSTM. Mais detalhes em [`rnn/README.md`](./rnn/README.md).

### 📚 Material Educacional e Atividades

- [`introduction.ipynb`](./introduction.ipynb): Notebook com exercícios de revisão de conceitos de machine learning.
- [`AM_Atividade_Avaliativa_1.ipynb`](./AM_Atividade_Avaliativa_1.ipynb): Atividade avaliativa sobre Machine Learning.

### ⚙️ Physics-Informed Neural Networks (PINNs)

- [`pinn_motor_thermal.ipynb`](./pinn/pinn_motor_thermal.ipynb): Implementação de Physics-Informed Neural Networks (PINNs) para resolver equações diferenciais parciais.
- [`README.md`](./pinn/README.md): Detalhes sobre a implementação e uso de PINNs.

### 🎓 Seminários

- **SVM (Support Vector Machines)**

  - [`svm.ipynb`](./seminar/svm/svm.ipynb): Notebook com a implementação e estudo de SVM.
  - [`README.md`](./seminar/svm/README.md): Detalhes específicos do seminário de SVM.

- **XGBoost**

  - [`xgboost.ipynb`](./seminar/xgboost/xgboost.ipynb): Notebook principal com a implementação e estudo de XGBoost.
  - [`XGBoost_exemplo.ipynb`](./seminar/xgboost/XGBoost_exemplo.ipynb): Exemplo prático de uso do XGBoost.
  - [`README.md`](./seminar/xgboost/README.md): Detalhes específicos do seminário de XGBoost.

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

### Predição de Temperaturas com LSTM

Uso de redes recorrentes do tipo LSTM para prever a temperatura mínima do dia seguinte em Melbourne. O projeto explora otimização de janelas temporais e hiperparâmetros.

### Estimativa de Temperatura em Motores com PINNs

Uso de Physics-Informed Neural Networks para prever a temperatura interna de motores elétricos a partir de medições de corrente e temperatura superficial. Projeto final do curso de Redes Neurais Artificiais e Deep Learning.

## 📝 Licença

Este projeto está licenciado sob os termos da licença incluída no arquivo [LICENSE](./LICENSE).
