import sys
from os import path

BASE_DIR = path.dirname(path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import numpy as np 
import pandas as pd 
import matplotlib as mt 
import unidecode as uni
from Scripts.utils import *
from Scripts.dataset import *

from Scripts.tfidf import *

MODELS_PATH = path.join(BASE_DIR, 'models')
list_intents = ['Hello', 'Inform', 'Request', 'feedback', 'Connect', 'Order', 'Changing', 'Return']
# list_intents = ['Hello']
intent_boundary = 6

def remove_symbol(sentences):
    for sentence in sentences:
        if sentence[0] == '@':
           sentence = sentence[intent_boundary:]
    return sentences

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
    svm_models = load_svm_models(list_intents, MODELS_PATH)
    tfidfconverter = make_tfidf_model(corpus=None, pretrained_path=path.join(MODELS_PATH,'model.pickle'))
    for i in range(len(sentences)):
        sentence = sentences[i]
        list_label = get_intent(list_intents, tfidfconverter, svm_models, sentence)
        df_data['labels'][i] = list_label
    return df_data

if __name__ == "__main__":
    df_data = classifier(data_file_path='data.txt')
    print(df_data)
