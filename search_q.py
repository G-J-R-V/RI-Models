import json
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from Models.vetorial import modelo_vetorial

### Global Stemmer
nltk.download("stopwords")

stemmer = SnowballStemmer("portuguese")
sw = set(stopwords.words("portuguese"))
###

### Global test_data
with open("./res/dados_loja.json", "r", encoding="utf-8") as file:
    test_data = json.load(file)

test_data_dict = {}

for i, value in enumerate(test_data):
    value["R"] = False
    test_data_dict[f"Doc{i}"] = value
###


def clean_query(query: str = "") -> list[str]:
    global stemmer
    global sw

    query = [
        stemmer.stem(w)
        for w in query.translate(str.maketrans("", "", string.punctuation)).split(" ")
        if w not in sw
    ]

    print("Clean query: " + f"{query}" + "\n")

    return query


def search_query(query: str = ""):

    global test_data_dict

    cleaned_query = clean_query(query)

    resultado_vetorial = modelo_vetorial(test_data_dict, cleaned_query)

    return resultado_vetorial  ##


if __name__ == "__main__":

    resultado = search_query("Blusa Azul Escuro")

    print(f"\nResultado: {resultado}")
