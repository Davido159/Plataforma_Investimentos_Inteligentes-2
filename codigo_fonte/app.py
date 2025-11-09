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

#Ignore future pandas/neuralprophet warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(layout="wide", page_title="Plataforma de Investimentos Inteligentes")
st.title("Plataforma de Investimentos Inteligentes 📈")
st.write("Previsão de Tendências com Machine Learning (NeuralProphet)") 

# RECOMMENDATION FUNCTION
def gerar_recomendacao(ganhos, gastos, tem_reserva):
    """
    Gera uma recomendação de investimento com base na renda, gastos e se o usuário
    já possui uma reserva de emergência.
    """
    saldo = ganhos - gastos

    # Level 1: Negative Balance
    if saldo <= 0:
        return (
            "🚨 **Nível 1 - Foco: Organização Financeira**\n\n"
            f"Seu saldo mensal é de **R$ {saldo:.2f}**. Neste momento, o foco principal não é investir, mas sim organizar as finanças.\n\n"
            "**Recomendação:** Conforme os princípios de educação financeira, o primeiro passo é revisar seu orçamento. "
            "Tente identificar onde é possível cortar gastos ou como aumentar sua renda para criar um saldo positivo.\n\n"
            "**Próximos Passos:**\n"
            "1.  Liste todas as suas despesas (fixas e variáveis).\n"
            "2.  Estabeleça um orçamento pessoal.\n"
            "3.  Se tiver dívidas, foque em estratégias para quitá-las."
        )
    
    # Level 2: Positive Balance, NO Reserve
    elif saldo > 0 and not tem_reserva:
        recomendacao = (
            f"🟢 **Nível 2 - Foco: Reserva de Emergência**\n\n"
            f"Parabéns! Você tem um saldo positivo de **R$ {saldo:.2f}** por mês.\n\n"
            "**Recomendação Principal:** Antes de pensar em ações (Renda Variável), seu primeiro e mais importante objetivo é construir sua **Reserva de Emergência**.\n\n"
            f"**O que é isso?** É um valor (geralmente de 3 a 6 meses de seus gastos mensais, ou seja, R$ {gastos*3:.2f} a R$ {gastos*6:.2f}) guardado para imprevistos.\n\n"
            "**Onde investir essa reserva?**\n"
            "Em investimentos de **Renda Fixa** com alta segurança e liquidez (que você possa sacar a qualquer momento):\n"
            "* Tesouro Direto (Ex: Tesouro Selic)\n"
            "* CDBs de grandes bancos que pagam 100% do CDI, com liquidez diária.\n\n"
            "--- \n"
            "**E a ferramenta de previsão de ações?**\n"
            "Use a ferramenta de previsão de ações para **estudar** e aprender. "
            "Quando sua reserva de emergência estiver completa, você estará pronto para o próximo nível."
        )
        return recomendacao

    # Level 3: Positive Balance, WITH Reserve
    elif saldo > 0 and tem_reserva:
        recomendacao = (
            f"🏆 **Nível 3 - Foco: Investimento (Renda Variável)**\n\n"
            f"Excelente! Você tem um saldo positivo de **R$ {saldo:.2f}** e sua reserva de emergência está completa.\n\n"
            "**Recomendação:** Você está no estágio ideal para começar a investir em **Renda Variável** (como ações) para fazer seu dinheiro crescer acima da inflação, conforme o objetivo da plataforma.\n\n"
            "**Próximos Passos:**\n"
            f"1.  Use a ferramenta **'2. Análise e Previsão de Ações'** aqui ao lado para analisar os ativos de baixa volatilidade que selecionamos.\n"
            f"2.  Considere investir uma *parte* do seu saldo (R$ {saldo:.2f}) todo mês nessas ações para construir seu patrimônio a longo prazo.\n"
            "3.  Lembre-se: Renda Variável envolve riscos. Nunca invista dinheiro que você possa precisar no curto prazo."
        )
        return recomendacao

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
    # Quantis adicionados para banda de incerteza
    modelo = NeuralProphet(quantiles=[0.05, 0.95]) 
    
    # Frequência 'B' (Business day) é importante
    modelo.fit(_dados_formatados, freq="B") 
    return modelo

# Sidebar

st.sidebar.header("1. Recomendação de Investimento")
ganhos = st.sidebar.number_input("Sua Renda Mensal (R$)", min_value=0.0, step=100.0, key="ganhos")
gastos = st.sidebar.number_input("Seus Gastos Mensais (R$)", min_value=0.0, step=100.0, key="gastos")

st.sidebar.caption("Marque a caixa abaixo se você já guardou o equivalente a 3-6 meses de seus gastos.")

tem_reserva = st.sidebar.checkbox("Já completei minha Reserva de Emergência")

btn_recomendacao = st.sidebar.button("Gerar Recomendação Pessoal")

st.sidebar.divider() 

# SECTION 2: Stock Analysis
st.sidebar.header("2. Análise e Previsão de Ações")

# Selection menu
lista_acoes_curadas = {
    "Banco do Brasil (BBAS3.SA)": "BBAS3.SA",
    "Itaú Unibanco (ITUB4.SA)": "ITUB4.SA",
    "Taesa (TAEE11.SA)": "TAEE11.SA",
    "Magazine Luiza (MGLU3.SA)": "MGLU3.SA",
    "Petrobras (PETR4.SA)": "PETR4.SA"
}
nome_amigavel = st.sidebar.selectbox("1. Selecione a Ação para Análise:", lista_acoes_curadas.keys())
codigo_input = lista_acoes_curadas[nome_amigavel] 

# Other settings
inicio = st.sidebar.date_input("2. Data de Início", pan.to_datetime("2015-01-01"))
fim = st.sidebar.date_input("3. Data de Fim", pan.to_datetime("2025-01-01"))
periodos_previsao = st.sidebar.slider("4. Período de Previsão (dia)", 30, 730, 365) 

btn_previsao = st.sidebar.button("Gerar Previsão de Ação")


# MAIN PANEL LOGIC

if btn_recomendacao:
    st.subheader("Recomendação de Investimento Pessoal")
    
    recomendacao = gerar_recomendacao(ganhos, gastos, tem_reserva)
    
    st.markdown(recomendacao)
    st.warning("⚠️ **Aviso:** Esta é uma recomendação educacional baseada nos princípios do projeto e não é uma garantia de retorno. Use como uma ferramenta de apoio à decisão.")

elif btn_previsao:
    
    # Data Download 
    with st.spinner(f"Baixando dados históricos para {codigo_input}..."):
        dados = baixar_dados(codigo_input, inicio, fim)
        if dados is None or dados.empty:
            st.error("Nenhum dado encontrado. Verifique sua conexão ou o Ticker da ação.")
            st.stop()
        
        st.subheader(f"Dados Históricos Brutos: {nome_amigavel}")
        st.dataframe(dados.tail())

    # Preprocessing
    dados_formatados = dados[['Close']].reset_index()
    dados_formatados.columns = ["ds", "y"]

    # Model Training
    with st.spinner("Treinando modelo de Machine Learning (NeuralProphet)... Isso pode levar alguns minutos."):
        modelo = treinar_modelo(dados_formatados)

    # Forecast Generation
    with st.spinner("Gerando previsões..."):
        previsoes_historicas = modelo.predict(dados_formatados)
        df_futuro_apenas = modelo.make_future_dataframe(dados_formatados, periods=periodos_previsao)
        previsoes_futuras_apenas = modelo.predict(df_futuro_apenas.tail(periodos_previsao)) 
        
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


    # Visualization  (Main Chart)
    st.subheader(f"Gráfico de Previsão de Preços para {periodos_previsao} dias")
    
    fig_forecast = plt.figure(figsize=(12, 6))
    plt.plot(dados_formatados["ds"], dados_formatados["y"], label="Histórico Real (2015-2024)", c="r")
    plt.plot(previsoes_historicas["ds"], previsoes_historicas["yhat1"], label="Previsão Histórica (Modelo)", c="b", linestyle="--")
    plt.plot(previsoes_futuras_apenas["ds"], previsoes_futuras_apenas["yhat1"], label=f"Previsão Futura ({periodos_previsao} dias)", c="g")
    
    # Adds the volatility/uncertainty band (quantiles)
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

    # Visualization (Components)
    st.subheader("Decomposição da Previsão (Tendência e Sazonalidade)")
    st.write("""
    Aqui vemos os padrões que o modelo aprendeu para fazer a previsão:
    - **Tendência (Trend):** A direção geral do preço (alta, baixa ou lateral).
    - **Sazonalidade (Seasonality):** Padrões que se repetem toda semana ou todo ano.
    """)
    
    fig_components = modelo.plot_components(previsoes_completas)
    st.plotly_chart(fig_components, use_container_width=True)
    
    # Table of Future Forecasts
    st.subheader("Dados Detalhados da Previsão Futura")
    cols_tabela = ['ds', 'yhat1', 'yhat1 5.0%', 'yhat1 95.0%', 'trend']
    st.dataframe(previsoes_futuras_apenas[cols_tabela].tail(15))

else:
    # Home screen (no buttons pressed)
    st.info("Bem-vindo! Use a barra lateral para gerar uma recomendação pessoal ou uma previsão de ação.")