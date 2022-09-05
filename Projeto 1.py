#!/usr/bin/env python
# coding: utf-8

# ## Primeiro treino e teste de um modelo de classificação (Porcos ou Cachorros?)

# In[ ]:





#  ## Definindo as Features

# In[17]:


# features (1 sim, 0 não)
# pelo longo? 
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1,1,1,0,0,0]


# In[18]:


from sklearn.svm import LinearSVC
model= LinearSVC()           # Chamo a Linear SVC que é uma classe que possibilita a criação de um modelo 
model.fit(dados, classes)   #passo o metodo fit para receber como parametro minhas duas variaveis (dados e classes)


# In[19]:


animal_misterioso = [1,1,1]        #aqui adicionei um animal misterioso que contem como 1 as features, usando o metodo predict() e o modelo me retornou [0] cachorro
model.predict([animal_misterioso])


# In[20]:


misterio1 = [1,1,1]            #aqui eu inclui mais alguns animais para pode testar o modelo novamente e trazer um arcuracia dos dados(algoritimo errou 1 dado)
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes = [misterio1, misterio2, misterio3]
previsoes = model.predict(testes)


# In[21]:


testes_classes = [0, 1, 1]
testes_classes


# In[22]:


previsoes == testes_classes     #comparando meus teste consigo ver que os dados estão errados quando o modelo calculo a predict


# In[23]:


from sklearn.metrics import accuracy_score  #usando a metrica de acuracia eu calculo as medias de testes e previões dos meus dados 

taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print("Taxa de acerto", taxa_de_acerto * 100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




