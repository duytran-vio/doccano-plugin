import pandas as pd
import os
import re
import json
import sys

from sklearn import svm,metrics
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from services.Scripts.entity import findall_index
from services.client import refresh_client, Client
from services.common import get_all_documents, build_label_map, get_documents
from services.evaluate import get_sequence_truth_doc, get_sequence

refresh_client()
doccano_client = Client.doccano_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'doccano_project_data')
DATA_CREATE_PATH = os.path.join(DATA_PATH, 'create')

intent_list = ['Hello', 'Done', 'Inform', 'Request', 'feedback', 'Connect', 'Order', 'Changing', 'Return']

tfidf_path = 'services/models/tfidf.pickle'
orig_data_path = 'services/models/orig_data'
orig_label_path = 'services/models/orig_label'
svm_path = 'services/models/hungne'

tfidfconverter = pickle.load(open(tfidf_path, 'rb'))
X_tfidf = pickle.load(open(orig_data_path, 'rb'))
label = pickle.load(open(orig_label_path, 'rb'))

def macro_f1(report):
    main_keys = list(report.keys())[0:-3]
    result = 0
    for i in main_keys: result += report[i]['f1-score']
    return result/len(main_keys)

### To retrain svm model base on original labels and new labels
### args: sents and their labels
def retrain(new_sents, new_labels):
    X_new = tfidfconverter.transform(new_sents).toarray()
    X = np.concatenate([X_new, X_tfidf], axis = 0)
    lbl = label + new_labels
    clf = svm.LinearSVC()
    clf = clf.fit(X, lbl)
    pickle.dump(X, open(orig_data_path, 'wb'))
    pickle.dump(lbl, open(orig_label_path, 'wb'))
    pickle.dump(clf, open(svm_path, 'wb'))

def get_id_start_end(file_name):
    '''
        Get project_id, start, end of file_name
        Argument: 
            file_name: str of file_name
        return:
            id, start, end
    '''
    sq = findall_index('\d+', file_name, 'none')
    if len(sq) < 3:
        return -1,-1,-1
    id = int(file_name[sq[0][0]: sq[0][1]])
    start = int(file_name[sq[1][0]: sq[1][1]])
    end = int(file_name[sq[2][0]: sq[2][1]])
    return id, start, end

def get_model_file(project_id, start, end):
    '''
        Get list of file that contain docs from start to end
    '''
    Files = os.listdir(DATA_CREATE_PATH)
    list_file = []
    first_start = start
    for File in Files:
        _id, L, R = get_id_start_end(File)
        if (project_id != _id or end < L or start > R):
            continue
        first_start = min(first_start, L)
        list_file.append(File)
    return list_file, first_start

def get_docs_from_file(file_name):
    '''
        return list of doc from start to end in file_name
    '''
    file_path = os.path.join(DATA_CREATE_PATH, file_name)
    json_list = list(open(file_path, 'r', encoding='utf-8'))
    docs = []
    for json_str in json_list:
        result = json.loads(json_str)
        docs.append(result)
    return docs

def get_model_docs(project_id, start, end):
    '''
        Return list of docs from start to end that was labeled by model
    '''
    list_file, first_start = get_model_file(project_id, start, end)
    if len(list_file) == 0:
        raise Exception("Model didn't label these documents")
    docs = []
    for File in list_file:
        file_docs = get_docs_from_file(File)
        docs.extend(file_docs)
    docs = docs[start - first_start: end - first_start + 1]
    return docs

def intent_to_num(ls_intent):
    if len(ls_intent) == 1:
        return intent_list.index(ls_intent[0])

def correct_label(project_id, start, end):
    '''
        Return: 
            result: list of dict
                [{
                    'text': str - document,
                    'label': int - index of intent
                }]
    '''
    model_docs = get_model_docs(project_id, start, end)
    correct_docs = get_documents(doccano_client, project_id, start - 1, end)
    labels_map = build_label_map(doccano_client, project_id)
    result = []
    new_sents, new_labels = [], []
    for i in range(len(correct_docs)):
        model_intents, _ = get_sequence_truth_doc(model_docs[i])
        correct_intents, _ = get_sequence(correct_docs[i], labels_map)
        text = correct_docs[i]['text']

        if model_intents != correct_intents and len(correct_intents) > 0:
            intent_num = intent_to_num(correct_intents)
            intent_summary = {
                'text': text,
                'label': intent_num
            }
            new_sents.append(text)
            new_labels.append(intent_num)
            result.append(intent_summary)
    retrain(new_sents, new_labels)

if __name__ == '__main__':
    correct_label(643, 3, 5)