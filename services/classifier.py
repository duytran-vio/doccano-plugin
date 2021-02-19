import sys
import os

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_path)

import numpy as np 
import pandas as pd 
import matplotlib as mt 
import unidecode as uni
from Scripts.utils import *
from Scripts.dataset import *

from Scripts.tfidf import *

mapping = {'272': 'Hello', 
            '273': 'Done', 
            '279': 'Inform',
            '280': 'Request',
            '294': 'Feedback',
            '274': 'Connect',
            '275': 'Order',
            '276': 'Changing',
            '277': 'Return'}
TRAIN_FILE = 'new5_55.csv'
CORPUS_FILE = 'data.txt'
intent_boundary = 6

def remove_symbol(df_corpus):
    for i in range(len(df_corpus)):
        if df_corpus['text'][i][0] == '@':
            df_corpus['text'][i] = df_corpus['text'][i][intent_boundary:]
    return df_corpus

def classifier(train_file_path, corpus_file_path):
    df_data, df_corpus_raw = read_from_file(data_path = train_file_path, \
                                    corpus_path = corpus_file_path)
    df_corpus_raw = df_corpus_raw.rename(columns = {0 : 'text'})
    df_corpus = df_corpus_raw.copy()
    df_corpus = remove_symbol(df_corpus)
    df_corpus_raw['labels'] = pd.np.empty((len(df_corpus_raw), 0)).tolist()

    for label_col in mapping.keys():

        data, label, corpus = data_from_file(part = df_data, full = df_corpus, \
                                    text_col = '0', label_col = label_col)

        tfidfconverter = make_tfidf_model(corpus, pretrained_path = 'services/model.pickle')
        X_corp_tfidf = tfidfconverter.transform(corpus).toarray()
        X_tfidf = tfidfconverter.transform(data).toarray()

        X_train, X_test, y_train, y_test, y_idx_train, y_idx_test =  svm_data_prepair(X_tfidf, label)

        clf = svm.SVC()
        clf = clf.fit(X_train, y_train)

        # y_pred = clf.predict(X_test)
        y_corp_pred = clf.predict(X_corp_tfidf)
        for i in range(len(y_corp_pred)):
            if df_corpus_raw['text'][i][0] == '@':
                if y_corp_pred[i] == 1:
                    df_corpus_raw['labels'][i].append(mapping[label_col])
            else:
                df_corpus_raw['labels'][i] = []

    # print('{:<15} {:<15} {:<25}'.format(mapping[label_col], len(data), \
    #                                     metrics.accuracy_score(y_pred, y_test) ) )
    return df_corpus_raw

if __name__ == "__main__":
    df_corpus_raw = classifier(train_file_path=TRAIN_FILE, corpus_file_path=CORPUS_FILE)
    for i in range(len(df_corpus_raw)):
        df_corpus_raw['labels'][i] = ','.join(df_corpus_raw['labels'][i])
    df_corpus_raw.to_excel('result.xlsx')
