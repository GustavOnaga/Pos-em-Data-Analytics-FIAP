import streamlit as st
import pandas as pd
import joblib

# ========================= Carregar modelo =========================

# modelo = joblib.load("modelo_rf.pkl")
modelo = joblib.load("Datathon/streamlit/modelo_rf.pkl")

st.title("Datathon: Fiap e Associação Passos Mágicos")
st.subheader("Modelo de Predição - Random Forest")
st.write("Insira os dados para gerar a previsão")

# ========================= Ano de ingresso =========================

Ano_ingresso = st.number_input("Ano de ingresso", value=2020, step=1)

st.divider()

# ========================= Métricas ================================

st.subheader("Métricas de desempenho")

col1, col2, col3 = st.columns(3)

with col1:
    INDE = st.number_input("INDE", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col2:
    IPV = st.number_input("IPV", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col3:
    IEG = st.number_input("IEG", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

col4, col5, col6 = st.columns(3)

with col4:
    IDA = st.number_input("IDA", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col5:
    IPP = st.number_input("IPP", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col6:
    IPS = st.number_input("IPS", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

st.divider()

# ========================= Notas  ==================================================

st.subheader("Notas")

portugues = st.slider(
    "Português",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

ingles = st.slider(
    "Inglês",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

matematica = st.slider(
    "Matemática",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

st.divider()

# ========================= Informações do aluno =========================

st.subheader("Informações do aluno")

col7, col8, col9 = st.columns(3)

with col7:
    idade = st.number_input("Idade", value=15.0)

with col8:
    IAA = st.number_input("Auto avaliação",  min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col9:
    genero = st.selectbox(
        "Gênero",
        options=[0, 1],
        format_func=lambda x: "Masculino" if x == 0 else "Feminino"
    )

st.divider()

# ========================= Botão de previsão =========================

if st.button("Realizar previsão"):

    dados = pd.DataFrame({
        "Ano_ingresso":[int(Ano_ingresso)],
        "IDA":[IDA],
        "IPV":[IPV],
        "IEG":[IEG],
        "genero":[genero],
        "matematica":[matematica],
        "portugues":[portugues],
        "ingles":[ingles],
        "IAA":[IAA],
        "IPS":[IPS],
        "idade":[idade],
        "INDE":[INDE],
        "IPP":[IPP]
    })

    previsao = modelo.predict(dados)

    st.subheader("Resultado")

    if previsao[0] == 1:
        st.markdown(
            """
            <div style="background-color:#ffcccc; padding:15px; border-radius:10px; font-size:18px;">
            ⚠️ <b>Probabilidade de apresentar defasagem.</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#ccffcc; padding:15px; border-radius:10px; font-size:18px;">
            ✅ <b>Sem probabilidade de apresentar defasagem.</b>
            </div>
            """,
            unsafe_allow_html=True
        )