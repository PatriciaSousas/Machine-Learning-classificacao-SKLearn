#!/usr/bin/env python
# coding: utf-8

# ### Analisar se usuários que acessaram determinadas páginas de um site irão ou não comprar um produto específico

# In[ ]:





# ## Importando Bibliotecas

# In[20]:


import pandas as pd


# In[ ]:





# ### Leitura dos dados

# In[21]:


url = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
dados = pd.read_csv(url)
dados 


# In[23]:


mapa ={                       #renomei as colunas do meu dataframe para portugues
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
    
}

dados=dados.rename(columns = mapa)
dados


# In[24]:


x= dados[["principal","como_funciona","contato"]] #separei os dados para treino e teste
y= dados[["comprou"]] 

x.head()


# ## Separando dados para Treino e Teste

# In[25]:


treino_x = x[:75]
treino_y = y[:75]                    #
teste_x = x[75:]
teste_y = y[75:]
teste_y.shape

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))


# In[28]:


from sklearn.svm import LinearSVC             #aplico treino + arcuracia
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)


# In[ ]:





# In[ ]:





# In[ ]:




