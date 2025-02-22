import pandas as pd
import json
import numpy as np
from tabulate import tabulate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def JsonToDataframe():
    with open("res/dados_loja.json", "r", encoding="utf-8") as file:
        test_data = json.load(file)

    colunas = ["Item"]

    df = pd.DataFrame(columns=colunas)

    for doc in test_data:
        nova_linha = {"Item": doc["Item"]}
        for word in doc["Descricao"]:
            if word == "dispon": # Quando chegar na seção de tamanhos ele pula para o próximo item
                break
            if word not in df.columns:  # Se uma palavra não estiver na coluna, ela é adicionada ao df e a linha recebe o valor 0
                df[word] = 0
            
            nova_linha[word] = 1 # Se a palavra estiver no df, ela é adicionada a nova linha e recebe o valor 1
        
        df = df._append(nova_linha,ignore_index=True)
    df = df.fillna(0) #troca o Nan por 0
    df = df.loc[:, df.nunique() > 1] #remove as colunas com valor constante
    

    return df

def KmeansSKL():
    df = JsonToDataframe()

    X = df.drop("Item", axis=1)

    kmeans = KMeans(n_clusters=4, random_state=42) # Cria um modelo KMeans com 4 clusters
    df["Cluster"] = kmeans.fit_predict(X)   # Encontra os melhores centros dos clusters e atribui a cada item um cluster    
    
    return df

def KMeansGroups():
    df = KmeansSKL()
    matriz = df.groupby("Cluster")["Item"].apply(list).tolist() # Agrupa os itens por cluster, Itens vão de 1 a 15
    print(matriz)

    print(df[["Item", "Cluster"]])

    return matriz


def ModifyJson():
    matriz = KMeansGroups()
    
    with open("res/dados_loja.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Criar um dicionário para mapear os itens
    item_dict = {item["Item"]: item for item in data}

    # Adicionar a similaridade para cada item
    for grupo in matriz:
        for item_id in grupo:   # Percorre o grupo de itens
            if item_id in item_dict: # Verifica se o item está no dicionário
                item_dict[item_id]["Similaridade"] = [id for id in grupo if id != item_id] 
                # Percorre o grupo de similares e adiciona o grupo de ids similares menos o id do item atual

    with open("res/dados_loja.json",  "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)