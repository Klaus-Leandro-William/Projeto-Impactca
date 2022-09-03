#!/usr/bin/env python
# coding: utf-8

# ## Instalando e Carregando os Pacotes

# In[1]:


# Imports

# Manipulação de dados
import pandas as pd
import numpy as np

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno

# Estatística
import scipy
from scipy.stats import normaltest
from scipy.stats import chi2_contingency

# Engenharia de Atributos
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce

# Ignore Warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ## Carregando os Dados

# In[2]:


# Carrega o dataset
df = pd.read_csv("C://Users//leandro.santos//Downloads//23-Projeto2//dataset//aug_train.csv")


# In[3]:


# Shape
df.shape


# In[4]:


# Colunas
df.columns


# In[5]:


# Amostra dos dados
df.head()


# In[6]:


# Info
df.info()


# ## Análise Exploratória de Dados

# In[7]:


# Descrevendo os dados não numéricos
df.describe(include = object)


# In[8]:


# Descrevendo os dados numéricos
df.describe().drop(columns = ['enrollee_id', 'target'])


# * Em **city_development_index** (CDI), os valores médios são 0,828, mediana 0,903 e std 0,123. Isso significa que a maioria dos candidatos é de cidades bem desenvolvidas.
# 
# 
# * Em **training_hours**, os valores médios são 65,367, mediana 47 e max 336. Isso significa que há mais candidatos com poucas horas de treinamento, mas alguns candidatos gastam muito tempo para fazer o treinamento.

# ### Visualizando as Variáveis Categóricas

# In[9]:


list(df.columns.values)[3:12]


# In[10]:


# Plot

# Tamanho da figura
plt.figure(figsize = (18,30))

# Lista de colunas
column_list = list(df.columns.values)[3:12]

# Contador
A = 0

# Loop
for i in column_list:
    A += 1
    plt.subplot(5, 2, A)
    ax = sns.countplot(data = df.fillna('NaN'), x = i)
    plt.title(i, fontsize = 15)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size = 12)
    if A >= 7:
        plt.xticks(rotation = 45)

# Layout
plt.tight_layout(h_pad = 2)


# In[ ]:





# In[ ]:





# In[ ]:




