def get_json():
    with open(r"res/loja_de_roupas_ultimos_itens.txt",'r') as arquivo:
        linhas = arquivo.readlines()
    
    for l in linhas:
        print(l)

get_json()