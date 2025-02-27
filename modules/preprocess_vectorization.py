"""preprocess_vectorization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BaGmA00Wk6-vhdoVCymRIYYEPf0MPINq
"""

!pip install num2words

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

data = pd.read_csv('/content/IMDB Dataset.csv')
data.head()

from preprocessing import *

def preprocess(data):
  data['review_new']=data['review'].apply(convert_lower_case)
  data['review_new']=data['review_new'].apply(remove_stop_words)
  data['review_new']=data['review_new'].apply(remove_punctuation)
  data['review_new']=data['review_new'].apply(remove_apostrophe)
  return data

data = data.iloc[:10]
preprocess(data)

"""### Tokenization"""

tokenized_data= data['review_new'].apply(lambda x: x.split())
tokenized_data.head()

"""### Stemming"""

stemmer= PorterStemmer()
stem_data = tokenized_data.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
stem_data.head()

"""### Lemmatization"""

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
 
# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()
lem_data = tokenized_data.apply(lambda x: [wnl.lemmatize(i) for i in x])
lem_data.head()

vocab_data = tokenized_data.apply(lambda x: set(x))
vocab_data.head()

"""### Vectorization

## TF-IDF

TF-IDF tends to Term Frequency and Inverse Document Frequency.

Term Frequency is the measure of the frequency of words in a document ie is the ratio of the number of times the word appears in a document compared to the total number of words in that document.

The Inverse Document Frequency is the log of the ratio of the number of documents to the number of documents containing the word.

### Implementation of the TF-IDF Model
"""

from tf_idf import *
length_document=count_words(vocab_data)
index_dict = ind_dict(vocab_data)
tf = TF(vocab_data,length_document,index_dict)
idf = IDF(index_dict,vocab_data)
tf_idf = TF_IDF(index_dict,tf,idf).T

df = pd.DataFrame(tf_idf)
df['label'] = data['sentiment']
df

from sklearn import preprocessing
 
label_encoder = preprocessing.LabelEncoder()
 
df['label']= label_encoder.fit_transform(df['label'])
df.head()

"""##  Frequency encoding"""

from count_frequency_encoding import *
count_frequency = count_frequency(vocab_data,length_document,index_dict).T

df_frequency_encoding = pd.DataFrame(count_frequency)
df_frequency_encoding['label'] = data['sentiment']

label_encoder = preprocessing.LabelEncoder()
df['label']= label_encoder.fit_transform(df['label'])
df.head()
