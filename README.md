## ☕ Coffee ML: Comparação de Modelos de Classificação

### 💡 Sobre o Projeto

O **Coffee ML** é um projeto de machine learning conceitual, desenvolvido para demonstrar e comparar diferentes métodos de classificação de dados.

O objetivo principal é ilustrar de forma clara como modelos teóricos se comportam na prática, permitindo ao usuário alterar hiperparâmetros do modelo e visualizar os resultados de treinamento e previsão.

### 🧠 Funcionalidades

O aplicativo permite:

1.  Carregamento de dados de treino.
2.  Seleção do modelo e seus hiperparâmetros.
3.  Treinamento e comparação de três métodos de classificação distintos:
    * **Regressão Logística (Logistic Regression - LR)**: Um modelo linear e robusto para classificação binária.
    * **Rede Neural Densa (Dense Neural Network - DNN)**: Um modelo não-linear, mais complexo, implementado via TensorFlow.
    * **Gradient Boosting por Árvores de Decisão (XGBoost - XGB)**: Um modelo sequencial que combina árvores de decisão.
4.  Visualização de resultados do treinamento (curva de aprendizado) e das previsões.

### 🛠️ Tecnologias e Bibliotecas

Este projeto utiliza o ecossistema Python para ML e visualização interativa:

| **Ferramenta** | **Objetivo** |
| :--- | :--- |
| Streamlit | Criação da interface web interativa. |
| TensorFlow / Keras | Construção e treinamento da Rede Neural Densa. |
| Scikit-learn | Implementação e treinamento da Regressão Logística. |
| XGBoost | Implementação do ensemble de Árvores de Decisão. |
| Pandas | Manipulação de dados tabulares. |
| NumPy | Operações vetoriais. |

### ℹ️ Como Executar Localmente

Siga os passos abaixo para rodar o aplicativo na sua máquina:

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/solardev-cs/coffee-ml.git](https://github.com/solardev-cs/coffee-ml.git)
    cd coffee-ml
    ```

2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use: .\venv\Scripts\activate
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  Inicie o aplicativo Streamlit:
    ```bash
    streamlit run app.py
    ```

O aplicativo será aberto automaticamente no seu navegador padrão.

Utilize os dados de treino e teste disponíveis na pasta \data.
