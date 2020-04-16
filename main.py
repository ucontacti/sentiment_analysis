import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

from functools import partial
import io
import os

import fastai
from fastai import *
from fastai.text import *

import seaborn as sns

df_features=pd.read_csv("Tweets.csv")


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms
# missingdata(df_features)
X = df_features

# X = df_features.drop("airline_sentiment",axis=1)
# y = df_features["airline_sentiment"]
# y = y.astype('category').cat.codes

# print(y.value_counts())


# preprocessing:
X["text"] = X["text"].str.replace("[^a-zA-Z]", " ")


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords 
stop_words = stopwords.words('english')

# tokenization 
tokenized_doc = X["text"].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization 
detokenized_doc = [] 
for i in range(len(X)): 
    t = ' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 

X["text"] = detokenized_doc
X_train, X_test, y_train, y_test = train_test_split(X, stratify = X["airline_sentiment"], test_size=0.30, random_state=42)
print(X_train.shape)
print(X_test.shape)

# Language model data
data_lm = TextLMDataBunch.from_df(train_df = X_train, valid_df = X_test, path = "")

# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = X_train, valid_df = X_test, vocab=data_lm.train_ds.vocab, bs=32)

learn = language_model_learner(data_lm,  arch = AWD_LSTM, pretrained = True, drop_mult=0.7)

learn.fit_one_cycle(1, 1e-2)