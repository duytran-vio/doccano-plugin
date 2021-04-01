import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split  
from sklearn import svm,metrics

import joblib

import re  
import unidecode

def equalize_label(part, col):
    cnt = 0
    max_cnt = len([i for i in range(part.shape[0]) if part[col][i]==1])
    drop_rows = []
    for i in range(part.shape[0]):
        if part[col][i] is None and cnt < 1.5*max_cnt:
            cnt+=1
        elif part[col][i] is None:
            drop_rows.append(i)
    return part.drop(part.index[drop_rows])

def eliminate_noise(doc):
    sentences = doc.split(' ')
    result = [i for i in sentences if len(i) <= 7]
    return ' '.join(result)

def preprocess(doc, remove_empty=False):
    results = []
 
    for i in range(len(doc)):  
        result = str(doc[i])

        # Remove all the special characters
        result = re.sub(r'\W', ' ', str(result))

        # Substituting multiple spaces with single space
        result = re.sub(r'\s+', ' ', result, flags=re.I)
        # Remove space at beginning and end
        result = re.sub('^\s', '', result)
        result = re.sub('\s$', '', result)
    
        # Converting to Lowercase
        result = result.lower()

        # result = unidecode.unidecode(result)
    
        if result == ' ' and remove_empty == True:
            continue
        
        #Remove customers' names
        result = re.sub('.*: ', '', result)

        results.append(result)

    return results