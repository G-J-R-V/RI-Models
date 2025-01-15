import json

import Models.vetorial
from Models.vetorial import modelo_vetorial

def search_query(query:str = ''):
    with open("res/dados_loja.json", "r") as file:
        test_data = json.load(file)

    return modelo_vetorial(test_data, query)


if __name__ == "__main__":
    search_query("Camisa vermelha")
