import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier



@st.cache_data
def load_data():
    df = pd.read_parquet('selectbox_posizioni.parquet')
    return df

df = load_data()



@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('catboost_posizioni.cbm', format='cbm')
    return model

model = load_model()



def predict(data):
    prediction = model.predict(data)
    return prediction



def main():
    st.title("Predittore posizioni lavorative")
    st.markdown(
    """
    Supporta ingestion di dati manuale e tramite file Excel o CSV.
    
    Per caricare un file Excel o CSV assicurarsi che contenga le seguenti colonne e in questo ordine
    - ruolo
    - funzione
    - azienda
    - Genere
    - studi
    - Età
    
    È case sensitive, per cui assicurarsi che i nomi delle colonne "Genere" e "Età" siano maiuscoli
    """)

    with st.form("manual_entry_form"):
        ruolo = st.selectbox("Seleziona il ruolo fra quelli presenti", options=df['ruolo'].unique())
        funzione = st.selectbox("Seleziona la funzione lavorativa fra quelle presenti", options=df['funzione'].unique())
        azienda = st.selectbox("Seleziona l'azienda fra quelle presenti", options=df['azienda'].unique())
        genere = st.selectbox("Genere", ["Maschile", "Femminile"])
        studi = st.selectbox("Seleziona gli studi fra quelli presenti", options=df['studi'].unique())

        eta = st.number_input("Età", min_value=0, max_value=100, step=1)
        submitted = st.form_submit_button("Submit")
        if submitted:

            data = pd.DataFrame([[ruolo, funzione, azienda, genere, studi, eta]],
                                columns=["ruolo", "funzione", "azienda", "Genere", "studi", "Età"])

            prediction = predict(data)
            st.write("Prediction:", prediction)


    uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["csv", "xlsx"])
    if uploaded_file is not None:

        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Prediction
        if st.button("Predict"):
            prediction = predict(data)
            st.write("Predictions:", prediction)

if __name__ == "__main__":
    main()
