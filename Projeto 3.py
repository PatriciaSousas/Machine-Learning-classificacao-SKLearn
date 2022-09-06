#!/usr/bin/env python
# coding: utf-8

# ## Aplicativo para fazer estimativas, por exemplo: o site demorará 42 horas para ser feito e quero pagar 275,00 reais

# In[22]:


import pandas as pd
get_ipython().system('pip install seaborn==0.9.0')


# In[23]:


dados = pd.read_csv('https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv')
dados


# In[24]:


a_renomear = {                                    #renomear colunas dentro do frame de horas
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
}
dados = dados.rename(columns = a_renomear)
dados.head()


# In[25]:


troca = {                               #inclui fazendo a troca os projetos que foram finalizados
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)
dados.head()


# In[26]:


sns.scatterplot(x="horas_esperadas", y= "preco", data=dados)  #plotei um dado para visualização basica dos dados


# In[28]:


sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data=dados) #separei e plotei os dois graficos finalizados e não finalizados para entender as horas minimas qu devem ser pags


# In[29]:


from sklearn.model_selection import train_test_split    #separei os dados testes e treino para estimular assertividade do meu projeto
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




