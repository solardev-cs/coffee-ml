import streamlit as st
import numpy as np
import locale
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier

# INICIALIZAÇÃO DE HIPERPARÂMETROS
if 'nn_neurons' not in st.session_state:
    st.session_state.nn_neurons = 5
if 'nn_activation' not in st.session_state:
    st.session_state.nn_activation = 'relu'
if 'nn_learning_rate' not in st.session_state:
    st.session_state.nn_learning_rate = 0.01
if 'nn_epochs' not in st.session_state:
    st.session_state.nn_epochs = 500

if 'lr_solver' not in st.session_state:
    st.session_state.lr_solver = 'liblinear'
if 'lr_C' not in st.session_state:
    st.session_state.lr_C = 1e9
if 'lr_max_iter' not in st.session_state:
    st.session_state.lr_max_iter = 1000

if 'xgb_estimators' not in st.session_state:
    st.session_state.xgb_estimators = 200
if 'xgb_learning_rate' not in st.session_state:
    st.session_state.xgb_learning_rate = 0.1
if 'xgb_max_depth' not in st.session_state:
    st.session_state.xgb_max_depth = 3

# CARREGAMENTO DE DADOS DE TREINO
def load_data(arquivo_upload):

    # Retorna None se nenhum arquivo foi carregado
    if arquivo_upload is None:        
        return None, None, None 

    # Carrega o arquivo usando Pandas
    df = pd.read_csv(arquivo_upload)
    
    # Validação de colunas
    required_cols = ['Temperatura (C)', 'Duracao (min)', 'Ideal (Y)']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"O CSV de treino deve conter as colunas: {required_cols}. "
            "Verifique se o arquivo está correto."
        )

    # Target deve ser tratado como inteiro para o Plotly
    df['Ideal (Y)'] = df['Ideal (Y)'].astype(int).astype(str)

    # Separa as Features (X) e o Target (Y)
    # X (Features): Temperatura e Duração
    X = df[['Temperatura (C)', 'Duracao (min)']].values    
    # Y (Target): Ideal
    Y = df[['Ideal (Y)']].astype(int).values
    
    return X, Y, df

# TREINAMENTO DOS MODELOS
@st.cache_data
def treina_modelo(X_train, Y_train, option, nn_neurons, nn_activation, nn_learning_rate, nn_epochs, lr_solver, lr_C, lr_max_iter, xgb_estimators, xgb_learning_rate, xgb_max_depth): 

    if option == "Rede Neural":        
        tf.random.set_seed(1234)

        # Modelo da rede neural
        model = keras.models.Sequential(
            [
                keras.Input(shape=(2,)),
                keras.layers.Dense(nn_neurons, activation=nn_activation, name = 'layer1'),
                keras.layers.Dense(1, activation='sigmoid', name = 'layer2')
            ]
        )
        
        # Compila o modelo
        model.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate=nn_learning_rate),
        )

        # Treinamento da rede
        history = model.fit(
            X_train,Y_train,            
            epochs=nn_epochs,
            batch_size=32,
            validation_split=0.2
        )

        st.success("✅ Treinamento concluído!")
        return model, history.history
    
    elif option == "Regressão Logística":

        # Modelo da regressão logística
        lr_model = LogisticRegression(
            solver=lr_solver,
            C=lr_C,
            max_iter=lr_max_iter,
            class_weight='balanced' 
        )

        # Treinamento do modelo
        lr_model.fit(X_train, Y_train)

        st.success("✅ Treinamento concluído!")
        return lr_model
    
    elif option == "XGBoost":

        # Modelo do ensemble de árvores de decisão
        xgb_model = XGBClassifier(
            n_estimators=xgb_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            objective='binary:logistic' 
        )

        # Treinamento do modelo
        xgb_model.fit(X_train, Y_train)

        st.success("✅ Treinamento concluído!")
        return xgb_model

def reseta_state():
    st.session_state.is_trained = False

# PORTUGUÊS DO BRASIL
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except locale.Error:
    print("Locale 'pt_BR.UTF-8' não disponível, utilizando o padrão do servidor.")

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Coffee ML",
    page_icon="☕️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "App que compara a qualidade preditiva de 3 modelos de ML para classificação binária."
    }
)

# MENU LATERAL
with st.sidebar:
    st.logo("images/logo.png", size="large", icon_image="images/icone.png")   
    st.write("")
    st.write("")
    st.subheader("⚙️ Configurações dos Modelos")
    option = st.selectbox(
        "Selecione o modelo de previsão:",
        ("Rede Neural", "Regressão Logística", "XGBoost"),
        on_change=reseta_state
    )
    st.write("")
    st.markdown("#### 📉 Hiperparâmetros")

    if option == "Rede Neural":    
        # Número de neurons da hidden layer
        st.session_state.nn_neurons = st.slider(
            "Número de Neurons (Hidden Layer)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Ajusta a complexidade da camada oculta."
        )
        # Tipo da função de ativação da hidden layer
        st.session_state.nn_activation = st.selectbox(
            "Função de Ativação (Hidden Layer)",
            options=["relu", "sigmoid", "tanh"],
            index=0, # 'relu'
            help="ReLU é o padrão moderno para otimização."
        )
        # Learning rate do modelo        
        st.session_state.nn_learning_rate = st.number_input(
            "Taxa de Aprendizado (Learning Rate)",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Ajusta o tamanho dos passos na otimização."
        )
        # Número de épocas de treinamento
        st.session_state.nn_epochs = st.number_input(
            "Épocas (Epochs)",
            min_value=5,
            max_value=5000,
            value=500,
            step=1,
            help="Número de vezes que o conjunto de dados completo será processado."
        )
        st.write("")
        st.markdown("#### ℹ️ Dados do Modelo")
        st.markdown(
            """
            - **Arquitetura:** MLP (Multi-Layer Perceptron)
            - **Camadas:** Input (2 features) -> Hidden -> Output (1 neuron)
            - **Output Layer:** `sigmoid`
            - **Função de Custo:** Binary Cross Entropy
            - **Otimizador:** Adam
            - **Biblioteca:** TensorFlow/Keras
            """
        )
        
    elif option == "Regressão Logística":
        # Tipo de solver utilizado
        st.session_state.lr_solver = st.selectbox(
            "Solver",
            options=["liblinear", "saga", "lbfgs"],
            index=0,
            help="Algoritmo de otimização."
        )        
        # Parâmetro C (inverso da força de regularização)
        st.session_state.lr_C = st.slider(
            "Força de Regularização (C)",
            min_value=0.1,
            max_value=1e9,
            value=1e9,
            step=100.0,
            help="Valor alto (1e9) = pouca regularização. Valor baixo (0.1) = forte regularização L2."
        )        
        # Número de iterações
        st.session_state.lr_max_iter = st.number_input(
            "Máximo de Iterações",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Limita o tempo para o solver convergir."
        )        
        st.write("")
        st.markdown("#### ℹ️ Dados do Modelo")
        st.markdown(
            """
            - **Arquitetura:** Modelo Linear
            - **Classificador:** Binário   
            - **Penalidade:** L2         
            - **Normalização:** `StandardScaler`
            - **Biblioteca:** Scikit-learn
            """
        )

    elif option == "XGBoost":
        # Número de estimadores
        st.session_state.xgb_estimators = st.number_input(
            "Número de Árvores",
            min_value=10,
            max_value=500,
            value=200,
            step=10,
            help="Define o número de árvores no ensemble."
        )     
        # Learning rate
        st.session_state.xgb_learning_rate = st.number_input(
            "Taxa de Aprendizado (Learning Rate)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            format="%.3f",
            help="Ajusta o tamanho dos passos na otimização."
        )      
        # Profundidade da árvore
        st.session_state.xgb_max_depth = st.slider(
            "Profundidade Máxima",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Profundidade máxima de cada árvore."
        )             
        st.write("")
        st.markdown("#### ℹ️ Dados do Modelo")
        st.markdown(
            """
            - **Arquitetura:** Gradient Boosted Decision Trees
            - **Algoritmo:** eXtreme Gradient Boosting  
            - **Função de Custo:** Log Loss         
            - **Biblioteca:** XGBoost
            """
        )       
    
    st.divider() 
    with st.container(horizontal=True):   
        st.space("large") 
        st.markdown(":grey[v2.0 (2025)  |  by CS]")        

# CABEÇALHO DO APP 
st.subheader("Coffee ML")
st.markdown("☕️ Aplicativo para prever a qualidade da torra de café.")
st.markdown("🧠 O app treina uma rede neural, uma regressão logística e um ensemble de árvores de decisão, classificando os dados de teste.")
st.divider()
st.markdown("#### Treino")

# PARTE CENTRAL DO APP
with st.container():

    # Inicializa session state para modelos
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'norm_l' not in st.session_state:
        st.session_state.norm_l = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    # Upload csv (dados de treinamento)
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:            
            arquivo_train = st.file_uploader(
                "Selecione o arquivo CSV com dados de treinamento (Temperatura, Duração, Ideal):", 
                type=['csv']
            )        

    # TREINAMENTO
    if arquivo_train is not None:
        try:
            X_train, Y_train, df = load_data(arquivo_train)
            
            if df is not None:
                st.info(f"Dados de treino carregados: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")

                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("")
                        st.write("")
                        st.dataframe(df, hide_index=True)
                    with col2:
                        # Gráfico dados de treino
                        fig = px.scatter(
                            df,
                            x='Temperatura (C)',
                            y='Duracao (min)',
                            color='Ideal (Y)', # Usa a coluna Ideal para colorir
                            color_discrete_map={'1': 'green', '0': 'red'},
                            labels={'1': 'Ideal (1)', '0': 'Não Ideal (0)'},
                            title='Temperatura vs. Duração'
                        )                        

                        fig.update_layout(legend_title_text='Torra Ideal') 

                        # Mostra o gráfico no Streamlit
                        st.plotly_chart(fig, width="stretch") 

                # Normalização da rede neural
                st.session_state.norm_l = keras.layers.Normalization(axis=-1)
                st.session_state.norm_l.adapt(X_train)  # aprende mediana e variância dos dados

                # Normalização da regressão logística    
                st.session_state.scaler = StandardScaler()

                # Treina rede neural
                if option == "Rede Neural":                    
                    Xn = st.session_state.norm_l(X_train)

                    # Aumenta tamanho do training set
                    # Isso substitui o uso do batch_size no modelo
                    #Xt = np.tile(Xn,(1000,1))
                    #Yt= np.tile(Y_train,(1000,1))

                    # Converte tensor em array numpy
                    Xt = Xn.numpy()
                    Yt = Y_train.astype(np.float32)                        

                    if st.button("Treinar Modelo NN"):
                        st.session_state.model, st.session_state.history = treina_modelo(
                            Xt, Yt, option, 
                            st.session_state.nn_neurons, 
                            st.session_state.nn_activation, 
                            st.session_state.nn_learning_rate, 
                            st.session_state.nn_epochs,
                            st.session_state.lr_solver, 
                            st.session_state.lr_C, 
                            st.session_state.lr_max_iter,
                            st.session_state.xgb_estimators,
                            st.session_state.xgb_learning_rate,
                            st.session_state.xgb_max_depth
                        )
                        st.session_state.is_trained = True                        

                # Treina regressão logística
                elif option == "Regressão Logística":
                    Xn = st.session_state.scaler.fit_transform(X_train)

                    # Converte targets para array numpy 1-D
                    Yt = Y_train.ravel()

                    if st.button("Treinar Modelo LR"):
                        st.session_state.model = treina_modelo(
                            Xn, Yt, option, 
                            st.session_state.nn_neurons, 
                            st.session_state.nn_activation, 
                            st.session_state.nn_learning_rate, 
                            st.session_state.nn_epochs,
                            st.session_state.lr_solver, 
                            st.session_state.lr_C, 
                            st.session_state.lr_max_iter,
                            st.session_state.xgb_estimators,
                            st.session_state.xgb_learning_rate,
                            st.session_state.xgb_max_depth
                        )
                        st.session_state.is_trained = True
                
                # Treina XGBoost
                elif option == "XGBoost":
                    Xn = X_train

                    # Converte targets para array numpy 1-D
                    Yt = Y_train.ravel()

                    if st.button("Treinar Modelo XGB"):
                        st.session_state.model = treina_modelo(
                            Xn, Yt, option, 
                            st.session_state.nn_neurons, 
                            st.session_state.nn_activation, 
                            st.session_state.nn_learning_rate, 
                            st.session_state.nn_epochs,
                            st.session_state.lr_solver, 
                            st.session_state.lr_C, 
                            st.session_state.lr_max_iter,
                            st.session_state.xgb_estimators,
                            st.session_state.xgb_learning_rate,
                            st.session_state.xgb_max_depth
                        )
                        st.session_state.is_trained = True
            
        except ValueError as e:
            st.error(f"Erro nos dados: {e}")
            st.session_state.is_trained = False
            st.session_state.model = None

    # PREVISÃO
    if st.session_state.is_trained and st.session_state.model is not None:
        st.markdown("✅ Modelo treinado e pronto para previsão!")
        st.divider()
        st.markdown("#### Previsão")

        # Parâmetros obtidos após treinamento da rede
        #W1, b1 = st.session_state.model.get_layer("layer1").get_weights()
        #W2, b2 = st.session_state.model.get_layer("layer2").get_weights()
        
        # Upload csv (dados de teste)
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                arquivo_test = st.file_uploader(
                    "Selecione o arquivo CSV com dados de teste (Temperatura, Duração):", 
                    type=['csv']
                )

        if arquivo_test is not None:
            try:
                # Carrega o CSV usando Pandas
                df_test = pd.read_csv(arquivo_test)
                
                st.info(f"Dados de teste carregados: {df_test.shape[0]} amostras.")

                # Pega as duas primeiras colunas
                X_test_df = df_test.iloc[:, 0:2] 
                # Converte para array numpy
                X_test = X_test_df.values

                if option == "Rede Neural":
                    # Normalização
                    X_testn = st.session_state.norm_l(X_test)
                    # Previsão    
                    predictions = st.session_state.model.predict(X_testn) 

                elif option == "Regressão Logística":
                    # Normalização
                    X_testn = st.session_state.scaler.transform(X_test)
                    # Previsão
                    predictions = st.session_state.model.predict_proba(X_testn)[:, 1]
                
                elif option == "XGBoost":
                    # Previsão    
                    predictions = st.session_state.model.predict_proba(X_test)[:, 1]

                # Cria colunas de probabilidade e previsão no dataframe
                df_test['Probabilidade (P)'] = np.round(predictions, decimals=4)
                df_test['Previsao (Y)'] = np.round(predictions).astype(int) 
                
                st.write("")
                st.write("")
    
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("🔶 **Resultados da Previsão**")
                        st.write("")
                        st.write("")

                        # Tabela de resultado com dados de teste
                        st.dataframe(df_test, hide_index=True)               
                
                        # Classificação binária
                        st.write(
                            "Amostras Classificadas como Ideal:", 
                            f"{df_test['Previsao (Y)'].sum()}"
                        )                        
                        
                    with col2:
                        st.markdown("🔶 **Resultados do Modelo**")

                        # Gráfico de dados de treino + teste
                        fig = px.scatter(
                            df_test,
                            x='Temperatura (C)',
                            y='Duracao (min)',
                            color='Previsao (Y)',
                            color_discrete_map={'1': 'green', '0': 'red'},
                            labels={'1': 'Ideal (1)', '0': 'Não Ideal (0)'},
                            title='Temperatura vs. Duração'
                        )                        

                        fig.update_layout(legend_title_text='Torra Ideal')

                        # Gráfico com fronteira de decisão para NN                        
                        if 'history' in st.session_state and option == "Rede Neural":                           

                            fig_nn = px.scatter(
                                df_test,
                                x='Temperatura (C)',
                                y='Duracao (min)',
                                color='Previsao (Y)', 
                                color_discrete_map={'1': 'green', '0': 'red'},
                                labels={'1': 'Ideal (1)', '0': 'Não Ideal (0)'},
                                title='Fronteira de Decisão NN'
                            )
                            fig_nn.update_layout(legend_title_text='Torra Ideal')
                            
                            # Cálculo da fronteira de decisão
                            model = st.session_state.model

                            # Define limites da plotagem
                            x1_min = df['Temperatura (C)'].min()
                            x1_max = df['Temperatura (C)'].max()
                            x2_min = df['Duracao (min)'].min()
                            x2_max = df['Duracao (min)'].max()

                            # Cria meshgrid (grade de 100x100 pontos)
                            x1_plot = np.linspace(x1_min, x1_max, 100)
                            x2_plot = np.linspace(x2_min, x2_max, 100)
                            X1, X2 = np.meshgrid(x1_plot, x2_plot)
                            X_grid = np.c_[X1.ravel(), X2.ravel()]

                            # Conversão do array numpy em tensor
                            X_grid_tensor = tf.constant(X_grid, dtype=tf.float32)
                            X_grid_normalized = st.session_state.norm_l(X_grid_tensor).numpy()

                            # Prever no grid (probabilidade para contornos suaves)
                            Z = model.predict(X_grid_normalized)
                            
                            # Mapear as previsões de volta para a forma 2-D (100, 100)
                            Z = Z.reshape(X1.shape)
                            
                            # Adiciona o contorno ao gráfico
                            fig_nn.add_contour(
                                x=x1_plot,
                                y=x2_plot,
                                z=Z,
                                showscale=False,
                                colorscale='RdBu',        
                                opacity=0.3,              
                                name='Fronteira de Decisão (NN)',
                                line_width=0,             
                                contours_coloring='fill', 
                                hoverinfo='skip'          
                            )

                            # Adiciona pontos de treino
                            fig_nn.add_scatter(
                                x=df['Temperatura (C)'],
                                y=df['Duracao (min)'],
                                mode='markers',
                                name='Dados de Treino',
                                marker=dict(
                                    color=df['Ideal (Y)'].map({'1': 'rgba(0, 128, 0, 0.4)', '0': 'rgba(255, 0, 0, 0.4)'}),
                                    size=8,
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                legendgroup='Treino',
                                showlegend=False
                            )
                            
                            # Ajusta a escala do eixo Y baseado no maior range total do conjunto de dados
                            y_min = min(df['Duracao (min)'].min(), df_test['Duracao (min)'].min())
                            y_max = max(df['Duracao (min)'].max(), df_test['Duracao (min)'].max())
                            margin = (y_max - y_min) * 0.05
                            fig_nn.update_yaxes(range=[y_min - margin, y_max + margin])

                            # Mostra o novo gráfico no Streamlit
                            st.plotly_chart(fig_nn, width="stretch")

                            # Gráfico da curva de aprendizado
                            st.markdown("**Curva de Aprendizado**")                            
                            hist_df = pd.DataFrame(st.session_state.history)                      
                            st.line_chart(hist_df[['loss', 'val_loss']]) 
                            #st.caption("A linha 'loss' deve cair consistentemente. Se 'val_loss' subir, há overfitting.") 

                        # Gráfico com fronteira de decisão para LR
                        if 'scaler' in st.session_state and option == "Regressão Logística":

                            # Obtém coeficientes e bias
                            W = st.session_state.model.coef_[0] # [W1, W2]
                            b = st.session_state.model.intercept_[0]
                            
                            mean = st.session_state.scaler.mean_
                            std = st.session_state.scaler.scale_
                        
                            # Desnormalização
                            W_orig = W / std
                            b_orig = b - np.sum(W * mean / std)
                            
                            # Cálculo da fronteira de decisão
                            # Equação no espaço original: W_orig[0]*x1 + W_orig[1]*x2 + b_orig = 0
                            # Isolando x2 (Duração): x2 = (-b_orig - W_orig[0] * x1) / W_orig[1]
                            
                            # Define o range de temperatura (x1) com base nos dados de treino
                            x1_min = df['Temperatura (C)'].min()
                            x1_max = df['Temperatura (C)'].max()
                            
                            X1_range = np.array([x1_min, x1_max])
                            
                            # Calcula a duração (x2) na fronteira
                            X2_boundary = (-b_orig - W_orig[0] * X1_range) / W_orig[1]
                            
                            # Dataframe da fronteira
                            df_boundary = pd.DataFrame({
                                'Temperatura (C)': X1_range,
                                'Duracao (min)': X2_boundary
                            })
                            
                            # Adiciona a fronteira ao gráfico
                            fig.add_scatter(
                                x=df_boundary['Temperatura (C)'],
                                y=df_boundary['Duracao (min)'],
                                mode='lines',
                                name='Fronteira de Decisão (LR)',
                                line=dict(color='yellow'),
                                showlegend=False
                            )
                            
                            fig.add_scatter(
                                x=df['Temperatura (C)'],
                                y=df['Duracao (min)'],
                                mode='markers',
                                name='Dados de Treino',
                                marker=dict(
                                    color=df['Ideal (Y)'].map({'1': 'rgba(0, 128, 0, 0.4)', '0': 'rgba(255, 0, 0, 0.4)'}),
                                    size=8
                                ),
                                legendgroup='Treino',
                                showlegend=False
                            )

                            fig.update_layout(title='Fronteira de Decisão LR')

                            # Define o range do eixo Y com base nos dados de treino
                            y_min = min(df['Duracao (min)'].min(), df_test['Duracao (min)'].min())
                            y_max = max(df['Duracao (min)'].max(), df_test['Duracao (min)'].max())
                            
                            # Margem para o gráfico
                            margin = (y_max - y_min) * 0.05
                            y_range_min = y_min - margin
                            y_range_max = y_max + margin                            
                            fig.update_yaxes(range=[y_range_min, y_range_max])

                            # Mostra o gráfico no Streamlit
                            st.plotly_chart(fig, width="stretch")

                        # Gráfico com fronteira de decisão para XGBoost
                        elif option == "XGBoost":
                            # Criação do gráfico base com os dados de teste
                            fig_xgb = px.scatter(
                                df_test,
                                x='Temperatura (C)',
                                y='Duracao (min)',
                                color='Previsao (Y)', 
                                color_discrete_map={'1': 'green', '0': 'red'},
                                labels={'1': 'Ideal (1)', '0': 'Não Ideal (0)'},
                                title='Fronteira de Decisão XGBoost'
                            )
                            fig_xgb.update_layout(legend_title_text='Torra Ideal')

                            # Recupera o modelo treinado
                            model = st.session_state.model

                            # Define limites da plotagem baseados nos dados de treino
                            x1_min, x1_max = df['Temperatura (C)'].min(), df['Temperatura (C)'].max()
                            x2_min, x2_max = df['Duracao (min)'].min(), df['Duracao (min)'].max()

                            # Cria meshgrid (grade de 100x100 pontos)
                            x1_plot = np.linspace(x1_min, x1_max, 100)
                            x2_plot = np.linspace(x2_min, x2_max, 100)
                            X1, X2 = np.meshgrid(x1_plot, x2_plot)
                            X_grid = np.c_[X1.ravel(), X2.ravel()]

                            # Prever no grid (Probabilidades)
                            Z = model.predict_proba(X_grid)[:, 1]
                            Z = Z.reshape(X1.shape)

                            # Adiciona o contorno ao gráfico
                            fig_xgb.add_contour(
                                x=x1_plot,
                                y=x2_plot,
                                z=Z,
                                showscale=False,
                                colorscale='RdBu',        
                                opacity=0.3,              
                                name='Fronteira XGBoost',
                                line_width=0,             
                                contours_coloring='fill', 
                                hoverinfo='skip'          
                            )

                            # Adiciona pontos de treino
                            fig_xgb.add_scatter(
                                x=df['Temperatura (C)'],
                                y=df['Duracao (min)'],
                                mode='markers',
                                name='Dados de Treino',
                                marker=dict(
                                    color=df['Ideal (Y)'].map({'1': 'rgba(0, 128, 0, 0.4)', '0': 'rgba(255, 0, 0, 0.4)'}),
                                    size=8,
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                showlegend=False
                            )

                            # Ajusta escala do eixo Y
                            y_min = min(df['Duracao (min)'].min(), df_test['Duracao (min)'].min())
                            y_max = max(df['Duracao (min)'].max(), df_test['Duracao (min)'].max())
                            margin = (y_max - y_min) * 0.05
                            fig_xgb.update_yaxes(range=[y_min - margin, y_max + margin])

                            # Mostra o gráfico final
                            st.plotly_chart(fig_xgb, width="stretch")                                                                             

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar o arquivo CSV: {e}")
                st.warning("Certifique-se de que o CSV possui as colunas de Temperatura e Duração.")
        
        else:
            st.info("Aguardando o upload do arquivo CSV para testar o modelo.")  

# RODAPÉ DO APP
#st.divider()
#with st.container():
#    col1,col2,col3 = st.columns([25,10,20])
#    with col2:            
#        st.markdown(":grey[v1.0 (2025)  |  by CS]")
                
            
                

    