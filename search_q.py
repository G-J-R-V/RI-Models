import json
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from Models.vetorial import modelo_vetorial

### Global Stemmer
nltk.download("stopwords")

stemmer = SnowballStemmer("portuguese")

# Removes stopwords
sw = set(stopwords.words("portuguese"))
###

### Global json_data
with open("res/dados_loja.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

json_data_dict = {}

for i, value in enumerate(json_data):
    value["R"] = False
    json_data_dict[f"Doc{i}"] = value
###


def clean_query(query: str = "") -> list[str]:
    global stemmer
    global sw

    query = [
        stemmer.stem(w)
        for w in query.translate(str.maketrans("", "", string.punctuation)).split(" ")
        if w not in sw
    ]

    print(f"Clean query: {query}")

    return query


def search_query(query: str = ""):

    global json_data_dict

    cleaned_query = clean_query(query)

    resultado_vetorial = modelo_vetorial(json_data_dict, cleaned_query)

    print(f"Resultado: {resultado_vetorial}\n")

    return resultado_vetorial


if __name__ == "__main__":

    resultado = search_query("Banana Azul")

    # print(f"\nResultado: {resultado}")
