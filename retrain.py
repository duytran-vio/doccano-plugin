from services.retrain import RETRAIN_PATH
import pandas as pd
import os
import json
import sys
import time
from datetime import datetime

from thundersvm import SVC


import smtplib
from email.mime.text import MIMEText

from sklearn import svm, metrics
import pickle
import numpy as np
from sklearn.metrics import classification_report

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


RETRAIN_PROJECT_PATH = 'services/retrain/retrain_project.csv'
X_train_path = 'services/retrain/X_train.txt'
X_test_path = 'services/retrain/X_test.txt'
y_train_path = 'services/retrain/y_train.pkl'
y_test_path = 'services/retrain/y_test.pkl'
f1_history_path = 'services/retrain/f1_history.pkl'
f1_logs_path = 'services/retrain/f1_logs.pkl'
svm_path = 'services/models/hungne'

accum_data_path = 'services/retrain/accum_data.txt'
accum_label_path = 'services/retrain/accum_label.txt'
new_datatest_path = 'services/retrain/new_data_test.txt'
new_labeltest_path = 'services/retrain/new_label_test.pkl'

tfidf_path = 'services/models/tfidf.pickle'
tfidfconverter = pickle.load(open(tfidf_path, 'rb'))

intent_list = ['Hello', 'Done', 'Inform', 'Order', 'Connect', 'feedback', 'Changing',\
            'Return', 'Other', 'Ok'] + ['Request_phone', 'Request_weight customer', None, \
            'Request_color_product','Request_cost_product', 'Request_shiping fee', \
            'Request_amount_product', 'Request_material_product', None, None, 'Request_size',\
            'Request_address', 'Request_product_image']


def write_text_to_file(file_path, data):
    f = open(file_path, 'w',encoding='utf-8')
    for i in data:
        f.write(str(i))
        f.write('\n')
    f.close()

def extract_f1(report):
    return report['macro avg']['f1-score'], report['weighted avg']['f1-score']

def read_vars():
    X_train = pd.read_csv(X_train_path, sep='\n', header=None).values[:,0]
    X_test = pd.read_csv(X_test_path, sep='\n', header=None).values[:,0]
    y_train = pickle.load(open(y_train_path, 'rb'))
    y_test = pickle.load(open(y_test_path, 'rb'))
    f1_history = pickle.load(open(f1_history_path, 'rb'))
    f1_logs = pickle.load(open(f1_logs_path, 'rb'))
    return X_train, X_test, y_train, y_test, f1_history, f1_logs

### Input: new sentences and labels
### If it can improve model -> apply and save
### Else send email to Bach to raise warning
### args: sents and their labels

def retrain(new_sents, new_labels):                     
    # for i in range(len(new_sents)):
    #     print(new_sents[i], intent_list[new_labels[i]])
    # return 

    ### Accumulate data until enough to start retraining
    try:
        accum_data = pd.read_csv(accum_data_path, sep='\n').values[:,0]
        accum_label = pd.read_csv(accum_label_path, sep='\n').values[:, 0]
        accum_label = [int(i) for i in accum_label]
    except:
        accum_data = []
        accum_label = []
    
    accum_data = np.concatenate([accum_data, new_sents], axis=0)
    accum_label = np.concatenate([accum_label, new_labels], axis=0)

    ### If data is not enough, save and wait, not train
    if len(accum_data) < 1000:
        write_text_to_file(accum_data_path, accum_data)
        write_text_to_file(accum_label_path, accum_label)
        print(len(accum_data) )
        print("NOT ENOUGH DATA, SAVE!")
        return None

    print(len(accum_data) )
    print("ENOUGH DATA, START RETRAINING")
    # for i in range(len(intent_list)):
    #     print(intent_list[i], ': ', np.count_nonzero(np.array(new_labels)==i))
    # print('------------------------------------')

    X_new = tfidfconverter.transform(accum_data).toarray()
    X_train_txt, X_test_txt, y_train, y_test, f1_his, f1_logs = read_vars()
    X_train, X_test = tfidfconverter.transform(X_train_txt).toarray(), tfidfconverter.transform(X_test_txt).toarray()

    X_train_txt = np.concatenate([accum_data, X_train_txt], axis = 0)

    X_train = np.concatenate((X_new, X_train), axis = 0)
    lbl = np.concatenate([accum_label,y_train], axis=0)
    # lbl = y_train

    # print(X_train.shape[0]) 
    # for i in range(len(intent_list)):
    #     print(intent_list[i], ': ', np.count_nonzero(lbl==i))
    # print('------------------------------------')

    clf = SVC(kernel='rbf', C=2, gamma=0.05)
    clf = clf.fit(X_train, lbl)

    y_pred = clf.predict(X_test)
    new_f1_logs = classification_report(y_test, y_pred, output_dict = False, digits=3,\
                    target_names=[intent_list[i] for i in range(0, len(intent_list)) if intent_list[i] is not None])
    new_report_f1 = classification_report(y_test, y_pred, output_dict = True,\
                    target_names=[intent_list[i] for i in range(0, len(intent_list)) if intent_list[i] is not None])
    new_f1 = extract_f1(new_report_f1)

    try:
        new_data_test = list(pd.read_csv(new_datatest_path, sep='\n', header=None).values[:,0])
        new_label_test = pickle.load(open(new_labeltest_path, 'rb'))
        X_new_test = tfidfconverter.transform(new_data_test).toarray()
        y_pred = clf.predict(X_new_test)
        tmp_labels = [i for i in range(len(intent_list)) if intent_list[i] != None]
        new_test_report_f1 = classification_report(new_label_test, y_pred, output_dict = True, labels = tmp_labels,\
                        target_names=[intent_list[i] for i in range(0, len(intent_list)) if intent_list[i] is not None]) 
        new_test_f1_logs = classification_report(new_label_test, y_pred, output_dict = False, digits=3, labels = tmp_labels,\
                        target_names=[intent_list[i] for i in range(0, len(intent_list)) if intent_list[i] is not None]) 
    except pd.errors.EmptyDataError:
        new_data_test, new_label_test = [], []
        new_test_report_f1 = {'macro avg':{'f1-score':0},'weighted avg':{'f1-score':0}}
        new_test_f1_logs = None
    new_test_f1 = extract_f1(new_test_report_f1)

    ### The data is stored in f1_history as a list, where
    ### each element is a 2-tuple (f1, f1_test)
    old_f1 = extract_f1(f1_his[-1][0])
    old_test_f1 = extract_f1(f1_his[-1][1])

    ### If the model is improved
    if new_f1[0]+0.015 >= old_f1[0] and new_f1[1]+0.015 >= old_f1[1] and \
        new_test_f1[0]+0.015 >= old_test_f1[0] and new_test_f1[1]+0.015 >= old_test_f1[1]:
        f1_his.append((new_report_f1, new_test_report_f1))
        f1_logs.append((new_f1_logs, new_test_f1_logs, str(datetime.now())  ))
        pickle.dump(f1_logs, open(f1_logs_path, 'wb'))
        pickle.dump(f1_his, open(f1_history_path, 'wb'))
        pickle.dump(lbl, open(y_train_path, 'wb'))
        pickle.dump(clf, open(svm_path, 'wb'))
        write_text_to_file(X_train_path, X_train_txt)
        write_text_to_file(accum_data_path, [])
        write_text_to_file(accum_label_path, [])

        new_data_test += new_sents
        new_label_test += new_labels
        write_text_to_file(new_datatest_path, new_data_test)
        pickle.dump(new_label_test, open(new_labeltest_path, 'wb'))
        print("SUCCESSFULLY UPDATE MODEL!")

    else: ### Raise warning
        with open('services/retrain/text.txt', 'rb') as fp:
            # Create a text/plain message
            msg = MIMEText(" ")
        # msg = MIMEText() 
        msg['Subject'] = 'WARNING FROM RETRAIN TMT CHATBOT'
        msg['From'] = "automessage.tmt@gmail.com"
        server.sendmail('automessage.tmt@gmail.com', 'ltb1002.edmail@gmail.com', msg.as_string())
        server.sendmail('automessage.tmt@gmail.com', 'hoalt@tmtsofts.com', msg.as_string())
        print("MODEL GET WORSE, WARNING MAIL SENT!")
    print("OLD f1:", old_f1)
    print("OLD TEST f1:", old_test_f1)
    print("NEW f1:", new_f1)
    print("NEW TEST f1", new_test_f1)
    # for k in new_report_f1.keys():
    #     print(k, ': ', new_report_f1[k])

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
    try:
        if len(ls_intent) == 1:
            return intent_list.index(ls_intent[0])
        else:
            intent = ls_intent[0] + '_' + ls_intent[1]
            return intent_list.index(intent)
    except:
        return -1

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
            if intent_num == -1: 
                continue
            intent_summary = {
                'text': text,
                'label': intent_num
            }
            new_sents.append(text)
            new_labels.append(intent_num)
            result.append(intent_summary)
    # print(result)
    retrain(new_sents, new_labels)

def use_retrain_model():
    df_project = pd.read_csv(RETRAIN_PROJECT_PATH)
    for i in range(len(df_project)):
        project = df_project.iloc[i]
        if not project['status']:
            correct_label(project['id'], project['start'], project['end'])
            df_project['status'][i] = True
    df_project.to_csv(RETRAIN_PROJECT_PATH, index = False)

def write_to_log():
    fi = open('retrain_logs.txt', 'a')
    now = datetime.now()
    t =  str(now.strftime('%Y-%m-%d %H:%M:%S')) + '\n'
    fi.write(t)

if __name__ == '__main__':
    ### setup to send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("automessage.tmt@gmail.com", "tmtpassword")
    use_retrain_model()
    write_to_log()
    # time.sleep(3600*24*7) #comment to debug