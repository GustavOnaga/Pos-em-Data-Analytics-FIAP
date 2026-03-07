import streamlit as st
import pandas as pd
import joblib

 
# ========================= Carregar modelo =========================
 
modelo = joblib.load("modelo_rf.pkl")

st.title("Modelo de Predição - Random Forest")

st.write("Insira os dados para gerar a previsão")


#  ========================= Inputs ========================= 

idade = st.number_input("Idade", value=15.0)

INDE = st.number_input("INDE", value=0.0)
Fase = st.number_input("Fase", value=0.0)
IPV = st.number_input("IPV", value=0.0)

matematica = st.number_input("Matemática", value=0.0)
ingles = st.number_input("Inglês", value=0.0)
IEG = st.number_input("IEG", value=0.0)
portugues = st.number_input("Português", value=0.0)

IDA = st.number_input("IDA", value=0.0)
IAA = st.number_input("IAA", value=0.0)
IPS = st.number_input("IPS", value=0.0)

Ano_ingresso = st.number_input("Ano de ingresso", value=2020, step=1)

genero = st.selectbox(
    "Gênero",
    options=[0,1],
    format_func=lambda x: "Masculino (0)" if x == 0 else "Feminino (1)"
)

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
        "Fase":[Fase]
    })

    previsao = modelo.predict(dados)

    st.subheader("Resultado")

    st.write(f"Predição do modelo: **{previsao[0]}**")