# sample code for GDA
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


def ind_dict(vocab_data):
  index_dict={}
  for i in range(len(vocab_data)):
    for word in vocab_data[i]:
      index_dict[word] = i
      i += 1
  return index_dict


def count_words(vocab_data):
  list_count=[]

  for i in range(len(vocab_data)):
    list_count.append(len(vocab_data[i]))

  return list_count


def count_frequency(vocab_data,length_document,index_dict):
  index_dict=list(index_dict.keys())
  result = np.zeros((len(index_dict),len(vocab_data)))

  for i in range(len(vocab_data)):
    for j in range(len(index_dict)):
      if index_dict[j] in vocab_data[i]:
        result[j,i]=1/length_document[i]

  return result
