import streamlit as st
import yfinance as yf
import pandas as pan
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import warnings

# Ignore future warnings from pandas/neuralprophet
warnings.simplefilter(action='ignore', category=FutureWarning)

#Page Configurationa
st.set_page_config(layout="wide", page_title="Plataforma de Investimentos Inteligentes")
st.title("Plataforma de Investimentos Inteligentes 📈")
st.write("Previsão de Tendências com Machine Learning (NeuralProphet)") 

# Cache Functions 
@st.cache_data
def baixar_dados(codigo, inicio, fim):
    try:
        dados = yf.download(codigo, start=inicio, end=fim, multi_level_index=False)
        return dados
    except Exception as e:
        return None

@st.cache_resource
def treinar_modelo(_dados_formatados):
    modelo = NeuralProphet(quantiles=[0.05, 0.95]) 
    
    modelo.fit(_dados_formatados, freq="B") 
    return modelo

# Sidebar (User Interface) 
st.sidebar.header("Configurações da Análise")

# Selection menu 
lista_acoes_curadas = {
    "Banco do Brasil (BBAS3.SA)": "BBAS3.SA",
    "Itaú Unibanco (ITUB4.SA)": "ITUB4.SA",
    "Taesa (TAEE11.SA)": "TAEE11.SA",
    "Magazine Luiza (MGLU3.SA)": "MGLU3.SA",
    "Petrobras (PETR4.SA)": "PETR4.SA"
}
nome_amigavel = st.sidebar.selectbox("1. Selecione a Ação para Análise:", lista_acoes_curadas.keys())
codigo_input = lista_acoes_curadas[nome_amigavel] # Converte o nome amigável no Ticker real

# Other settings
inicio = st.sidebar.date_input("2. Data de Início", pan.to_datetime("2015-01-01"))
fim = st.sidebar.date_input("3. Data de Fim", pan.to_datetime("2025-01-01"))
periodos_previsao = st.sidebar.slider("4. Período de Previsão (dias)", 30, 730, 365) 

if st.sidebar.button("Gerar Previsão"):
    
    # Data Donwload
    with st.spinner(f"Baixando dados históricos para {codigo_input}..."):
        dados = baixar_dados(codigo_input, inicio, fim)
        if dados is None or dados.empty:
            st.error("Nenhum dado encontrado. Verifique sua conexão ou o Ticker da ação.")
            st.stop()
        
        st.subheader(f"Dados Históricos Brutos: {nome_amigavel}")
        st.dataframe(dados.tail())

    #  pre-processing 
    dados_formatados = dados[['Close']].reset_index()
    dados_formatados.columns = ["ds", "y"]

    #  Model Training 
    with st.spinner("Treinando modelo de Machine Learning (NeuralProphet)... Isso pode levar alguns minutos."):
        modelo = treinar_modelo(dados_formatados)

    # Forecast Generation
    with st.spinner("Gerando previsões..."):
        previsoes_historicas = modelo.predict(dados_formatados)
        df_futuro_apenas = modelo.make_future_dataframe(dados_formatados, periods=periodos_previsao)
        previsoes_futuras_apenas = modelo.predict(df_futuro_apenas)
        previsoes_completas = pan.concat([previsoes_historicas, previsoes_futuras_apenas])
        
    # EVALUATION AND INTERPRETATION
    st.subheader("🎯 Confiabilidade da Previsão (Tradução)")
    
    y_true = dados_formatados['y']
    y_pred = previsoes_historicas['yhat1'] 
    
    if len(y_true) == len(y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nível de Aderência:", f"{r2*100:.1f}%")
            if r2 > 0.90:
                st.success("**Interpretação:** Excelente. O modelo acertou a tendência histórica com alta precisão.")
            elif r2 > 0.70:
                st.info("**Interpretação:** Bom. O modelo conseguiu seguir bem a tendência geral dos preços.")
            elif r2 > 0.50:
                st.warning("**Interpretação:** Razoável. O modelo acertou a direção, mas teve dificuldades com as variações.")
            else:
                st.error("**Interpretação:** Baixo. O modelo teve dificuldade em prever os preços com precisão.")
        
        with col2:
            st.metric("Erro Médio da Previsão:", f"R$ {mae:.2f}")
            st.info(f"""
            **Interpretação:** Em média, quando o modelo previu os preços históricos, ele errou em **R$ {mae:.2f}** (para mais ou para menos) em relação ao preço real.
            (Isso equivale a um erro médio de **{mape:.2f}%**.)
            """)
            
    else:
        st.error(f"Erro na avaliação: Inconsistência de dados (Reais: {len(y_true)}, Previstos: {len(y_pred)})")

    # Investment Signal
    st.subheader(f"Sinal de Tendência para {periodos_previsao} dias")

    ultimo_preco_real = dados_formatados['y'].iloc[-1]
    ultimo_preco_previsto = previsoes_futuras_apenas['yhat1'].iloc[-1]
    percentual_mudanca = ((ultimo_preco_previsto - ultimo_preco_real) / ultimo_preco_real) * 100

    col_sinal, col_desc = st.columns(2)
    with col_sinal:
        if percentual_mudanca > 5.0: 
            st.metric(label="Sinal do Modelo", value="🟢 Tendência de ALTA")
        elif percentual_mudanca < -5.0: 
            st.metric(label="Sinal do Modelo", value="🔴 Tendência de BAIXA")
        else:
            st.metric(label="Sinal do Modelo", value="🟡 Tendência LATERAL")

    with col_desc:
        st.markdown(f"""
        - **Último Preço Real (em {fim.strftime('%d/%m/%Y')}):** R$ {ultimo_preco_real:.2f}
        - **Previsão para {periodos_previsao} dias:** R$ {ultimo_preco_previsto:.2f}
        - **Variação Prevista:** {percentual_mudanca:.2f}%
        """)
    st.warning("⚠️ **Aviso:** Este é um modelo preditivo baseado em dados históricos e não é uma garantia de retorno. Use como uma ferramenta de apoio à decisão.")


    # Visualization - Main Chart
    st.subheader(f"Gráfico de Previsão de Preços para {periodos_previsao} dias")
    
    fig_forecast = plt.figure(figsize=(12, 6))
    plt.plot(dados_formatados["ds"], dados_formatados["y"], label="Histórico Real (2015-2024)", c="r")
    plt.plot(previsoes_historicas["ds"], previsoes_historicas["yhat1"], label="Previsão Histórica (Modelo)", c="b", linestyle="--")
    plt.plot(previsoes_futuras_apenas["ds"], previsoes_futuras_apenas["yhat1"], label=f"Previsão Futura ({periodos_previsao} dias)", c="g")
    
    # Add the volatility band
    plt.fill_between(
        previsoes_completas["ds"],
        previsoes_completas["yhat1 5.0%"],
        previsoes_completas["yhat1 95.0%"],
        color="g",
        alpha=0.2,
        label="Intervalo de Incerteza (90%)"
    )
    
    plt.legend()
    plt.title(f"Previsão de {nome_amigavel}")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento (R$)")
    st.pyplot(fig_forecast)

    # Visualization - Components)
    st.subheader("Decomposição da Previsão (Tendência e Sazonalidade)")
    st.write("""
    Aqui vemos os padrões que o modelo aprendeu para fazer a previsão:
    - **Tendência (Trend):** A direção geral do preço (alta, baixa ou lateral).
    - **Sazonalidade (Seasonality):** Padrões que se repetem toda semana ou todo ano.
    """)
    
    fig_components = modelo.plot_components(previsoes_completas)
    st.plotly_chart(fig_components, use_container_width=True)
    
    # Forecast Table Visualization
    st.subheader("Dados Detalhados da Previsão Futura")
    cols_tabela = ['ds', 'yhat1', 'yhat1 5.0%', 'yhat1 95.0%', 'trend']
    st.dataframe(previsoes_futuras_apenas[cols_tabela].tail(15))

else:
    st.info("Bem-vindo! Por favor, selecione uma ação na barra lateral e clique em 'Gerar Previsão'.")