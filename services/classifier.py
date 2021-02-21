import sys
from os import path

BASE_DIR = path.dirname(path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import json
import numpy as np 
import pandas as pd 
import matplotlib as mt 
import unidecode as uni
from Scripts.utils import *
from Scripts.dataset import *

from Scripts.tfidf import *

MODELS_PATH = path.join(BASE_DIR, 'models')
list_intents = ['Hello', 'Inform', 'Request', 'feedback', 'Connect', 'Order', 'Changing', 'Return']
list_entities = ['ID_product', 'size_product',
       'color_product', 'material_product', 'cost_product', 'amount_product',
       'name_promotion', 'content_promotion', 'Id member', 'phone member',
       'addr member', 'level member', 'benefit member', 'feedback',
       'shiping fee', 'size customer', 'height customer', 'weight customer',
       'addr store', 'phone store']

### load svm models as a dictionary from files
def load_svm_models(intent_list, input_path='/content/svm_models/'):
    result = {}
    for intent in intent_list:
        filename = path.join(input_path, intent)
        model = pickle.load(open(filename, 'rb'))
        result[intent] = model
    return result

### return the list of intents of sentence
### ### intent_list: list of intents to predict
### ### tfidfconverter: TfidfVectorizer of sklearn.feature_extraction.text
### ### svm_models: a dict of SVM models pretrained for each intent
### ### sentence: single sentence to work with
def get_intent(intent_list, tfidfconverter, svm_models, sentence):
    result = []
    sent_tfidf = tfidfconverter.transform([sentence]).toarray()
    for intent in intent_list:
        if svm_models[intent].predict(sent_tfidf) == 1:
            result.append(intent)
    return result

def classifier(data_file_path):
    sentences, df_data = corpus_from_file(corpus_path = data_file_path)
    df_data = df_data.rename(columns = {0 : 'text'})
    df_data['labels'] = np.empty((len(df_data), 0)).tolist()

    ### label intent
    svm_models = load_svm_models(list_intents, MODELS_PATH)
    tfidfconverter = make_tfidf_model(corpus=None, pretrained_path=path.join(MODELS_PATH,'model.pickle'))
    for i in range(len(sentences)):
        sentence = sentences[i]
        list_intent_label = get_intent(list_intents, tfidfconverter, svm_models, sentence)
        n_intents = 0
        ls_intents = []
        for label in list_intent_label:
            ls_intents.append([n_intents, n_intents + 1, label])
            n_intents = n_intents + 1
        df_data['labels'][i] = ls_intents

    ### label entity
    vocab_file = open(path.join(MODELS_PATH,'vocab.json'))
    vocab = json.load(vocab_file)
    for i in range(len(df_data)):
        sentence = df_data['text'][i]
        list_entity_label = get_entity(list_entities, vocab, sentence)
        list_entity_label = remove_duplicate_entity(list_entity_label, len(sentence))
        df_data['labels'][i].extend(list_entity_label)

    return df_data
    
### return labels: a list of [start, end, entity] of all enities input
### ### entity_list: list of entity 
### ### vocab: output of make_vocab()
def get_entity(entity_list, vocab, sentence):
    sentence = sentence.lower()
    labels = []
    for entity in entity_list:
       labels += infer(entity, vocab[entity], sentence)
    return labels

### return a list of [start, end, entity]
def infer(entity, vocab, sentence):
    result = []
    for phrase in vocab:
        list_match_idx = find_substring_Z(phrase, sentence)
        for idx in list_match_idx:
            result.append([idx, idx+len(phrase), entity])
    return result

def find_substring_Z(sub, s):
    if len(sub) == 0: 
      return []
    st = sub + '.' + s
    z = [0] * len(st)
    i = 1
    L = 0
    R = 0
    while i < len(st):
        if L <= i and i <= R:
            z[i] = min(z[i - L], R - i)
        while i + z[i] < len(st) and st[z[i]] == st[i + z[i]]:
            z[i] = z[i] + 1
        if (i + z[i] > R):
            L = i
            R = i + z[i]
        i = i + 1
    idx = []
    for i in range(len(st)):
        if z[i] == len(sub) and (len(idx) == 0 or idx[-1] + len(sub) <= i):
            idx.append(i)
    idx = [x - len(sub) - 1 for x in idx]
    return idx

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
    df_data = classifier(data_file_path='data.txt')
    print(df_data)
