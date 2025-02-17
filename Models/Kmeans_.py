import pandas as pd
import json
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

    print(kmeans.labels_)
    
    
    return df

    

df = KmeansSKL()

print(df[["Item", "Cluster"]])