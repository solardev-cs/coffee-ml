import streamlit as st
import numpy as np
import locale
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# INICIALIZAÇÃO DE HIPERPARÂMETROS
if 'nn_neurons' not in st.session_state:
    st.session_state.nn_neurons = 5
if 'nn_activation' not in st.session_state:
    st.session_state.nn_activation = 'relu'
if 'nn_learning_rate' not in st.session_state:
    st.session_state.nn_learning_rate = 0.01
if 'nn_epochs' not in st.session_state:
    st.session_state.nn_epochs = 10

if 'lr_solver' not in st.session_state:
    st.session_state.lr_solver = 'liblinear'
if 'lr_C' not in st.session_state:
    st.session_state.lr_C = 1e9
if 'lr_max_iter' not in st.session_state:
    st.session_state.lr_max_iter = 1000

def load_data(arquivo_upload):
    """ 
    Carrega os dados de treino de um arquivo CSV.
    
    Args:
        uploaded_train_file: Objeto de arquivo carregado (UploadFile do Streamlit).
        
    Returns:
        X: Array numpy de features (Temperatura, Duracao).
        Y: Array numpy do alvo (Ideal).
    """
    if arquivo_upload is None:
        # Retorna None se nenhum arquivo foi carregado
        return None, None, None 

    # 1. Carrega o arquivo usando Pandas
    df = pd.read_csv(arquivo_upload)
    
    # Validação básica de colunas (ajuste estes nomes conforme seu CSV real)
    required_cols = ['Temperatura (C)', 'Duracao (min)', 'Ideal (Y)']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"O CSV de treino deve conter as colunas: {required_cols}. "
            "Verifique se o arquivo está correto."
        )

    # Garante que o alvo (Y) seja tratado como categórico/inteiro para o Plotly
    df['Ideal (Y)'] = df['Ideal (Y)'].astype(int).astype(str)

    # 2. Separa as Features (X) e o Target (Y)
    # X (Features): Temperatura e Duração
    X = df[['Temperatura (C)', 'Duracao (min)']].values
    
    # Y (Target): Ideal
    Y = df[['Ideal (Y)']].astype(int).values

    # 3. Converte os DataFrames para Arrays NumPy (formato esperado pelo Keras)
    #X = X_df.values
    #Y = Y_df.values
    
    return X, Y, df

@st.cache_resource
def treina_modelo(X_train, Y_train, option, nn_neurons, nn_activation, nn_learning_rate, nn_epochs, lr_solver, lr_C, lr_max_iter): 

    if option == "Rede Neural":
        # Modelo da rede neural
        tf.random.set_seed(1234)  # applied to achieve consistent results

        model = keras.models.Sequential(
            [
                keras.Input(shape=(2,)),
                keras.layers.Dense(nn_neurons, activation=nn_activation, name = 'layer1'),
                keras.layers.Dense(1, activation='sigmoid', name = 'layer2')
            ]
        )
        
        # Define rede neural
        model.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate=nn_learning_rate),
        )

        # Treinamento da rede neural
        history = model.fit(
            X_train,Y_train,            
            epochs=nn_epochs,
            batch_size=32,
            validation_split=0.2
        )

        st.success("✅ Treinamento concluído!")
        return model, history.history
    
    elif option == "Regressão Logística":
        # Força o solver liblinear e desliga a regularização
        # max_iter=1000 garante tempo suficiente para convergir
        lr_model = LogisticRegression(
            solver=lr_solver,
            C=lr_C,
            max_iter=lr_max_iter,
            class_weight='balanced' 
        )

        lr_model.fit(X_train, Y_train)

        st.success("✅ Treinamento concluído!")
        return lr_model

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
        'About': "App que compara a qualidade preditiva de uma rede neural vs uma regressão logística clássica."
    }
)

# MENU LATERAL
with st.sidebar:
    st.write("")
    st.write("")
    st.subheader("⚙️ Configurações dos Modelos")
    option = st.selectbox(
        "Selecione o modelo de previsão:",
        ("Rede Neural", "Regressão Logística"),
        on_change=reseta_state
    )
    st.write("")
    st.markdown("#### 📉 Hiperparâmetros")

    # GUIA 1: REDE NEURAL (NN)
    if option == "Rede Neural":    
        # Hiperparâmetros Otimizáveis
        st.session_state.nn_neurons = st.slider(
            "Número de Neurons (Hidden Layer)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Ajusta a complexidade da camada oculta."
        )
        st.session_state.nn_activation = st.selectbox(
            "Função de Ativação (Hidden Layer)",
            options=["relu", "sigmoid", "tanh"],
            index=0, # 'relu'
            help="ReLU é o padrão moderno para otimização."
        )        
        st.session_state.nn_learning_rate = st.number_input(
            "Taxa de Aprendizado (Learning Rate)",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Ajusta o tamanho dos passos na otimização."
        )
        st.session_state.nn_epochs = st.number_input(
            "Épocas (Epochs)",
            min_value=5,
            max_value=5000,
            value=10,
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
        
    # GUIA 2: REGRESSÃO LOGÍSTICA (LR)
    elif option == "Regressão Logística":
        st.session_state.lr_solver = st.selectbox(
            "Solver",
            options=["liblinear", "saga", "lbfgs"],
            index=0, # 'liblinear' (o mais usado por você)
            help="Algoritmo de otimização."
        )        
        # Parâmetro C (Inverso da força de Regularização)
        st.session_state.lr_C = st.slider(
            "Força de Regularização (C)",
            min_value=0.1,
            max_value=1e9,
            value=1e9,
            step=100.0,
            help="Valor alto (1e9) = pouca regularização. Valor baixo (0.1) = forte regularização L2."
        )        
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
        
st.logo("images/logo.png", size="large", icon_image="images/icone.png")     
st.subheader("Coffee ML")
st.markdown("☕️ Aplicativo para prever a qualidade da torra de café.")
st.markdown("🧠 O app treina uma rede neural simples e uma regressão logística, classificando os dados de teste.")
st.divider()
st.markdown("#### Treino")

# PARTE CENTRAL DO APP
with st.container():

    # Inicializa Session State do app
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

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            # Upload csv (dados de treinamento)
            arquivo_train = st.file_uploader(
                "Selecione o arquivo CSV com dados de treinamento (Temperatura, Duração, Ideal):", 
                type=['csv']
            )        

    # Carregar os dados de treino quando o arquivo for enviado
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
                        # Gráfico de Dispersão (Scatter Plot)
                        fig = px.scatter(
                            df,
                            x='Temperatura (C)',
                            y='Duracao (min)',
                            # Usa a coluna 'Ideal (Y)' para colorir
                            color='Ideal (Y)',
                            # Define rótulos amigáveis para a legenda
                            color_discrete_map={'1': 'green', '0': 'red'},
                            labels={'1': 'Ideal (1)', '0': 'Não Ideal (0)'},
                            title='Temperatura vs. Duração'
                        )                        
                        # Atualiza o layout para melhor visualização (opcional)
                        fig.update_layout(legend_title_text='Torra Ideal')                        
                        # Mostra o gráfico no Streamlit
                        st.plotly_chart(fig, width="stretch") 

                # Normalização da rede neural
                st.session_state.norm_l = keras.layers.Normalization(axis=-1)
                st.session_state.norm_l.adapt(X_train)  # learns mean, variance

                # Normalização da regressão logística    
                st.session_state.scaler = StandardScaler()

                if option == "Rede Neural":                    
                    Xn = st.session_state.norm_l(X_train)

                    # Aumenta tamanho do training set
                    # Isso substitui o uso do batch_size no modelo
                    #Xt = np.tile(Xn,(1000,1))
                    #Yt= np.tile(Y_train,(1000,1))

                    # Converte tensor em array numpy
                    Xt = Xn.numpy()
                    Yt = Y_train.astype(np.float32)                        

                    # O botão e o treinamento ficam condicionados à presença dos dados
                    if st.button("Treinar Modelo NN"):
                        st.session_state.model, st.session_state.history = treina_modelo(
                            Xt, Yt, option, 
                            st.session_state.nn_neurons, 
                            st.session_state.nn_activation, 
                            st.session_state.nn_learning_rate, 
                            st.session_state.nn_epochs,
                            st.session_state.lr_solver, 
                            st.session_state.lr_C, 
                            st.session_state.lr_max_iter
                        )
                        st.session_state.is_trained = True                        

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
                            st.session_state.lr_max_iter
                        )
                        st.session_state.is_trained = True
            
        except ValueError as e:
            st.error(f"Erro nos dados: {e}")
            st.session_state.is_trained = False
            st.session_state.model = None

    # --- Lógica Pós-Treinamento ---
    if st.session_state.is_trained and st.session_state.model is not None:
        st.markdown("✅ Modelo treinado e pronto para previsão!")
        st.divider()
        st.markdown("#### Previsão")

        # Parâmetros obtidos após treinamento da rede
        #W1, b1 = st.session_state.model.get_layer("layer1").get_weights()
        #W2, b2 = st.session_state.model.get_layer("layer2").get_weights()
        #st.dataframe(W1, hide_index=True)
        #st.write("W2:\n", W2, "\nb2:", b2)
        
        # Exemplo de uso:
        # 1. Widget de Upload de Arquivo
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                arquivo_test = st.file_uploader(
                    "Selecione o arquivo CSV com dados de teste (Temperatura, Duração):", 
                    type=['csv']
                )
        # 2. Processar o arquivo se ele foi carregado
        if arquivo_test is not None:
            try:
                # Carregar o CSV usando Pandas
                df_test = pd.read_csv(arquivo_test)
                
                st.info(f"Dados de teste carregados: {df_test.shape[0]} amostras.")
                
                # 3. Preparação dos Dados (Assumindo que as 2 primeiras colunas são as features)
                # Adapte as colunas conforme o nome/posição real do seu CSV de teste
                
                # Se você usou o nome das colunas da Opção 1:
                if 'Temperature (C)' in df_test.columns and 'Duration (min)' in df_test.columns:
                    X_test_df = df_test[['Temperature (C)', 'Duration (min)']]
                # Caso contrário, use as 2 primeiras colunas (índice 0 e 1):
                else:
                    X_test_df = df_test.iloc[:, 0:2] # Pega as duas primeiras colunas

                # Converter para array numpy, que é o formato esperado pelo Keras/TensorFlow
                X_test = X_test_df.values

                # Normalização dos dados de previsão
                if option == "Rede Neural":
                    # Normalização
                    X_testn = st.session_state.norm_l(X_test)
                    # Previsão    
                    predictions = st.session_state.model.predict(X_testn)                 
                elif option == "Regressão Logística":
                    # Normalização
                    X_testn = st.session_state.scaler.transform(X_test)
                    # Previsão
                    probabilities = st.session_state.model.predict_proba(X_testn)[:, 1] 
                    predictions = probabilities

                # 5. Apresentar os Resultados
                # Adiciona as previsões de volta ao DataFrame para visualização
                df_test['Probabilidade (P)'] = np.round(predictions, decimals=4)
                #threshold = 0.2
                #df_test['Previsao (Y)'] = (df_test['Probabilidade (P)'] >= threshold).astype(int)
                df_test['Previsao (Y)'] = np.round(predictions).astype(int) 
                
                st.write("")
                st.write("")
    
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("🔶 **Resultados da Previsão**")
                        st.write("")
                        st.write("")
                        st.dataframe(df_test, hide_index=True)               
                
                        # Se o alvo for a classificação (0 ou 1)
                        st.write(
                            "Amostras Classificadas como Ideal:", 
                            f"{df_test['Previsao (Y)'].sum()}"
                        )                        
                        
                    with col2:
                        st.markdown("🔶 **Resultados do Modelo**")
                        # Gráfico de Dispersão (Scatter Plot)
                        fig = px.scatter(
                            df_test,
                            x='Temperatura (C)',
                            y='Duracao (min)',
                            # Usa a coluna 'Ideal (Y)' para colorir
                            color='Previsao (Y)',
                            # Define rótulos amigáveis para a legenda
                            color_discrete_map={'1': 'green', '0': 'red'},
                            labels={'1': 'Ideal (1)', '0': 'Não Ideal (0)'},
                            title='Temperatura vs. Duração'
                        )                        
                        # Atualiza o layout para melhor visualização (opcional)
                        fig.update_layout(legend_title_text='Torra Ideal')

                        if 'history' in st.session_state and option == "Rede Neural":                           
                            # --- 0. Inicialização do Gráfico ---
                            # Cria o gráfico inicial com os dados de teste (usando 'fig_nn' para não misturar)
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
                            
                            # --- 1. Cálculo do Meshgrid e Normalização (USANDO O Keras norm_l) ---
                            model = st.session_state.model

                            # Definir limites do plot (baseado nos dados de treino/gerais)
                            x1_min = df['Temperatura (C)'].min()
                            x1_max = df['Temperatura (C)'].max()
                            x2_min = df['Duracao (min)'].min()
                            x2_max = df['Duracao (min)'].max()

                            # Criar Meshgrid (grade de 100x100 pontos)
                            x1_plot = np.linspace(x1_min, x1_max, 100)
                            x2_plot = np.linspace(x2_min, x2_max, 100)
                            X1, X2 = np.meshgrid(x1_plot, x2_plot)
                            X_grid = np.c_[X1.ravel(), X2.ravel()] # Dados em formato (N, 2)

                            # CORREÇÃO: Aplicar a normalização do Keras (norm_l)
                            # Convertemos o array numpy para tensor antes de aplicar a camada
                            X_grid_tensor = tf.constant(X_grid, dtype=tf.float32)
                            X_grid_normalized = st.session_state.norm_l(X_grid_tensor).numpy() # Aplica a camada e volta para numpy
                            
                            # --- 2. Previsão e Contorno ---

                            # Prever no grid (probabilidade para contornos suaves)
                            Z = model.predict(X_grid_normalized)
                            
                            # Mapear as previsões de volta para a forma 2D (100, 100)
                            Z = Z.reshape(X1.shape)
                            
                            # Adicionar o Contorno (Fronteira de Decisão) ao Gráfico
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

                            # --- 3. Adicionar Pontos de Treino e Finalizar ---

                            # Adicionar os Pontos de TREINO (Para ficarem por cima do Contorno)
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
                            
                            # Ajustar a escala do Eixo Y (baseado no range total)
                            y_min = min(df['Duracao (min)'].min(), df_test['Duracao (min)'].min())
                            y_max = max(df['Duracao (min)'].max(), df_test['Duracao (min)'].max())
                            margin = (y_max - y_min) * 0.05
                            fig_nn.update_yaxes(range=[y_min - margin, y_max + margin])

                            # Mostrar o novo gráfico (fig_nn)
                            st.plotly_chart(fig_nn, width="stretch")

                            st.markdown("**Curva de Aprendizado**")                            
                            hist_df = pd.DataFrame(st.session_state.history)                      
                            st.line_chart(hist_df[['loss', 'val_loss']]) 
                            st.caption("A linha 'loss' deve cair consistentemente. Se 'val_loss' subir, há overfitting.") 

                        if 'scaler' in st.session_state and option == "Regressão Logística":
                            # 1. Obter Coeficientes e Intercepto
                            W = st.session_state.model.coef_[0] # [W1, W2]
                            b = st.session_state.model.intercept_[0]
                            
                            mean = st.session_state.scaler.mean_
                            std = st.session_state.scaler.scale_
                        
                            # 3. Desnormalização (Conversão dos coeficientes para a escala original)
                            W_orig = W / std
                            b_orig = b - np.sum(W * mean / std)
                            
                            # 4. Cálculo da Linha de Decisão (Fronteira)
                            # Equação no espaço original: W_orig[0]*x1 + W_orig[1]*x2 + b_orig = 0
                            # Isolando x2 (Duração): x2 = (-b_orig - W_orig[0] * x1) / W_orig[1]
                            
                            # Definir o Range de Temperatura (x1) com base nos dados de treino (df)
                            x1_min = df['Temperatura (C)'].min()
                            x1_max = df['Temperatura (C)'].max()
                            
                            X1_range = np.array([x1_min, x1_max])
                            
                            # Calcular a Duração (x2) na fronteira
                            X2_boundary = (-b_orig - W_orig[0] * X1_range) / W_orig[1]
                            
                            # 5. Criar o DataFrame da Fronteira
                            df_boundary = pd.DataFrame({
                                'Temperatura (C)': X1_range,
                                'Duracao (min)': X2_boundary
                            })
                            
                            # 6. Adicionar a Fronteira ao Gráfico
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

                            # 7. Atualizar o Título do Gráfico
                            fig.update_layout(title='Fronteira de Decisão LR')

                            # 1. Definir o range do Eixo Y com base nos dados de treino (df)
                            y_min = min(df['Duracao (min)'].min(), df_test['Duracao (min)'].min())
                            y_max = max(df['Duracao (min)'].max(), df_test['Duracao (min)'].max())
                            
                            # 2. Adicionar uma margem (5% do total) para o gráfico não ficar grudado nos pontos min/max
                            margin = (y_max - y_min) * 0.05
                            y_range_min = y_min - margin
                            y_range_max = y_max + margin
                            
                            # 3. Aplicar o novo range ao Eixo Y
                            fig.update_yaxes(range=[y_range_min, y_range_max]) # <--- A Linha Mágica

                            # Mostra o gráfico (isso substitui a chamada original do gráfico)
                            st.plotly_chart(fig, width="stretch")                                                                             

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar o arquivo CSV: {e}")
                st.warning("Certifique-se de que o CSV possui as colunas de Temperatura e Duração.")
        
        else:
            st.info("Aguardando o upload do arquivo CSV para testar o modelo.")  

# RODAPÉ DO APP
with st.container():
    st.write("")
    st.divider()
    with st.container():
        col1,col2,col3 = st.columns([25,10,20])
        with col2:            
            st.markdown(":grey[v1.0 (2025)  |  by CS]")
                
            
                

    