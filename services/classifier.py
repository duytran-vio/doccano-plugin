import sys
from os import path

BASE_DIR = path.dirname(path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import json
import numpy as np 
import pandas as pd 
import matplotlib as mt 
import unidecode as uni
import joblib
from Scripts.utils import *
from Scripts.dataset import *

from Scripts.tfidf import *
from Scripts.entity import label_entity
import re

import time

import concurrent.futures

MODELS_PATH = path.join(BASE_DIR, 'models')
# list_intents = ['Hello', 'Inform', 'Request', 'feedback', 'Connect', 'Order', 'Changing', 'Return', 'Done']
# list_intents = ['Hello', 'Inform', 'Request', 'feedback', 'Connect', 'Order'] ### remove Return and Changing
intent_list = ['Hello', 'Done', 'Inform', 'Order', 'Connect', 'feedback', 'Changing', 'Return', 'Other', \
    'OK', 'Request_phone', 'Request_weight customer', 'Request_height customer', 'Request_color_product','Request_cost_product', \
    'Request_shiping fee', 'Request_amount_product', 'Request_material_product', None, None, 'Request_size', 'Request_address',\
    'Shop_wrong', 'Client_wrong', 'Request_product_image']
# intent_list = ['hello', 'done', 'inform', 'order', 'connect', 'feedback', 'changing', 'return', 'other',\
#     'ok', 'phone', 'weight customer', 'height customer', 'color_product', 'cost_product', \
#     'shiping fee', 'amount_product', 'material_product', None, None, 'size', 'address', 'product_image']

### resolve later:
### ### need rules-based method: 'size_product', 'amount_product', 'size customer', 'phone members', \
### ### 'addr member', 'shipping fee', 'size customer', 'height customer', 'weight customer', 'addr store', \
### ### 'feedback', 'phone store'
### ### to few values: 'name_promotion', 'content_promotion', 'level member', 'benefit member'

# list_entities = ['ID_product', 'size_product',
#        'color_product', 'material_product', 'cost_product', 'amount_product',
#        'name_promotion', 'content_promotion', 'Id member', 'phone member',
#        'addr member', 'level member', 'benefit member', 'feedback',
#        'shiping fee', 'size customer', 'height customer', 'weight customer',
#        'addr store', 'phone store']

list_entities = ['ID_product', 'material_product']

### load svm models as a dictionary from files
def load_svm_models(intent_list, input_path='/content/svm_models/'):
    start_time = time.time()
    result = {}
    for intent in intent_list:
        filename = path.join(input_path, intent)
        # model = pickle.load(open(filename, 'rb'))
        model = joblib.load(filename + '.joblib')
        result[intent] = model
    print("Load all svm models in ", time.time()-start_time, 's.')
    return result

### return the prediction for an array of sentences with respect to a specific intent
### ### intent: intent to predict
### ### svm_models: a dict of SVM models pretrained for each intent
### ### sent_tfidf: sentences transformed to TFIDF
def worker(intent, svm_models, sent_tfidf):
    prediction = svm_models[intent].predict(sent_tfidf)
    return np.concatenate([prediction, np.array([intent])])

def worker(sent_tfidf, svm_models):
    return svm_models.predict(sent_tfidf)

def load_svm_multiclass_models(input_path='/content/svm_models/'):
    
    filename = path.join(input_path,'hungne')
    # model = joblib.load(filename+'.joblib')
    model = pickle.load(open(filename, 'rb'))
    
    return model

def classifier(data_file_path, address_inp):
    print('Start classifying')

    sentences, df_data = corpus_from_file(corpus_path = data_file_path)
    df_data = df_data.rename(columns = {0 : 'text'})
    df_data['labels'] = np.empty((len(df_data), 0)).tolist()

    # svm_models = load_svm_models(list_intents, MODELS_PATH)
    start_time = time.time()
    svm_models = load_svm_multiclass_models(MODELS_PATH)
    tfidfconverter = make_tfidf_model(corpus=None, pretrained_path=path.join(MODELS_PATH,'tfidf.pickle'))
    sent_tfidf = tfidfconverter.transform(sentences).toarray()

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(worker, intent,svm_models, sent_tfidf) for intent in list_intents]

    # results = [i.result() for i in results]

    # for i in range(len(sentences)):
    #     sentence = sentences[i]
    #     list_intent_label = [results[j][-1] for j in range(len(list_intents)) if float(results[j][i])==1]
    #     # list_intent_label = get_intent(list_intents, tfidfconverter, svm_models, sentence)
    #     n_intents = 0
    #     ls_intents = []
    #     for label in list_intent_label:
    #         ls_intents.append([n_intents, n_intents + 1, label])
    #         n_intents = n_intents + 1
    #     df_data['labels'][i] = ls_intents
    #     # if i % 100 == 0: print(i)

    sents_intent = worker(sent_tfidf, svm_models)
    for i in range(len(sentences)):
        if re.search('.*khách.*:', df_data['text'][i].lower()) is not None: 
            # df_data['labels'][i] = [[0, 1, intent_list[int(sents_intent[i])]]]
            intent = intent_list[int(sents_intent[i])]
            if intent.find('Request_') >= 0:
                entity = intent[intent.find('_') + 1: ]
                ls_intent = [[0, 1, 'Request'], [1, 2, entity]]
            else:
                ls_intent = [[0, 1, intent]]
            df_data['labels'][i] = ls_intent
        

    end_time_intent = time.time()
    print('Get all intents in ', end_time_intent-start_time, 's.')


    ### label entity
    start_time = time.time()
    sents_entity = label_entity(df_data['text'].tolist(), address_inp)
    for i in range(len(df_data)):
        df_data['labels'][i].extend(sents_entity[i])

    end_time_entity = time.time()
    print('Get all entities in ', end_time_entity - start_time, 's.')    

    return df_data

def get_intent(intent_list, tfidfconverter, svm_models, sentence):
    result = []
    sent_tfidf = tfidfconverter.transform([sentence]).toarray()
    for intent in intent_list:
        if svm_models[intent].predict(sent_tfidf) == 1:
            result.append(intent)
    return result

def remove_duplicate_entity(sent_entities, sent_len):
    a = [x for x in range(len(sent_entities))]
    a.sort(key = lambda x: sent_entities[x][1] - sent_entities[x][0], reverse=True)
    check = [False] * (sent_len + 1)
    final_entities = []
    for x in a:
        seq_label = sent_entities[x]
        if check[seq_label[0]] or check[seq_label[1]]: 
            continue
        for i in range(seq_label[0], seq_label[1]):
            check[i] = True
        final_entities.append(seq_label)
    return final_entities

if __name__ == "__main__":
    df_data = classifier(data_file_path='data.txt',address_inp= None)
    # df_data = classifier(data_file_path='data_3k.txt')
    print(df_data)