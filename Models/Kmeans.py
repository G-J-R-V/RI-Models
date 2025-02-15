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
            '''if word == "dispon":
                break'''
            if word not in df.columns:
                df[word] = 0
            
            nova_linha[word] = 1
        
        df = df._append(nova_linha,ignore_index=True)
    df = df.fillna(0) #troca o Nan por 0
    df = df.loc[:, df.nunique() > 1] #remove as colunas com valor constante

    return df

def KmeansSKL():
    df = JsonToDataframe()

    X = df.drop("Item", axis=1)

    kmeans = KMeans(n_clusters = 5,random_state = 0).fit(X)

    print(kmeans.labels_)

    

KmeansSKL()