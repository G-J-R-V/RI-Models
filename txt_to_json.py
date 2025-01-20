import json
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

def get_json():
    with open(r"res/loja_de_roupas_ultimos_itens.txt",'r',encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()

    lista = []
    d = {
        'Item':'',
        'Nome':'',
        'Descricao':'',
        'Preco':''
    }
    ctrl = 0
    for l in linhas:
        if ctrl==0:
            d['Item'] = l.replace('Item ','').replace('\n','')
        elif ctrl==1:
            d['Nome'] = l.replace("Nome: ",'').replace('\n','')
        elif ctrl==2:
            d['Descricao'] = l.replace('Descrição: ','').replace('\n','')
        else:
            d['Preco'] = float(l.replace('Preço: R$ ','').replace(',','.').replace('\n',''))

        ctrl+=1
        if ctrl>=4:
            lista.append(d.copy())
            ctrl=0
    
    with open(r'res/dados_loja.json','w',encoding='utf-8') as arquivo_json:
        json.dump(lista,arquivo_json,indent=4,ensure_ascii=False)

def tratar_desc():
    stemmer=SnowballStemmer("portuguese")
    sw= set(stopwords.words('portuguese'))
    with open(r"res/dados_loja.json","r",encoding='utf-8') as arquivo:
        dados=json.load(arquivo)

    for i in dados:
        l = []
        lista = i['Descricao'].lower().replace('.','').replace(',','').split()
        lista=[w for w in lista if not w in sw]
        for p in lista:
            l.append(stemmer.stem(p))
        i['Descricao']=l
    
    with open(r'res/dados_loja.json','w',encoding='utf-8') as arquivo_json:
        json.dump(dados,arquivo_json,indent=4,ensure_ascii=False)


get_json()
tratar_desc()