import numpy as np
import os
import csv
import pandas as pd
from doccano_api_client import DoccanoClient

doccano_client: DoccanoClient = None
pre_link = 'http://103.113.81.36:8000/projects/'

RETRAIN_FILE = 'retrain_project.csv'
FINISH_FILE = 'finish_project.xlsx'

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'doccano_project_data')
DATA_SAMPLE_PATH = os.path.join(DATA_PATH, 'sample')
DATA_DOWNLOAD_PATH = os.path.join(DATA_PATH, 'download')
RETRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'retrain')

def handle_request(request, client: DoccanoClient):
    global doccano_client
    doccano_client = client
    project_id, start, end = extract_request(request)
    RETRAIN_PROJECT_FILE = os.path.join(RETRAIN_PATH, RETRAIN_FILE)
    retrain_project = {
        'id': project_id,
        'start': start,
        'end': end,
        'status': False
    }
    
    exist = os.path.exists(RETRAIN_PROJECT_FILE)
        
    fi = open(RETRAIN_PROJECT_FILE, 'a', newline='')
    fields = retrain_project.keys()
    writer = csv.DictWriter(fi, fieldnames=fields)
    if not exist: writer.writeheader()
    writer.writerow(retrain_project)
    add_finish(retrain_project)

def add_finish(project):
    FINISH_FILE_PATH = os.path.join(RETRAIN_PATH, FINISH_FILE)
    try:
        df = pd.read_excel(FINISH_FILE_PATH, engine='openpyxl')
    except:
        df = pd.DataFrame(columns=['id', 'name', 'finish', 'link'])
    index = [i for i in range(len(df)) if df['id'][i] == np.int64(project['id'])]
    fdocs = get_fdocs_from_index(index, df)
    fdocs.append([project['start'], project['end']])
    fdocs = merge(fdocs)
    add_df = to_df(fdocs, project['id'])
    df = df.drop(index)
    df = df.append(add_df, ignore_index=True)
    df['id'] = df['id'].astype(int)
    df = df.sort_values(by=['id'], ignore_index=True)
    df.to_excel(FINISH_FILE_PATH, index = False, engine='xlsxwriter')

def get_fdocs_from_index(index, df):
    fdocs = []
    for i in index:
        fdoc = to_int(df['finish'][i])
        fdocs.append(fdoc)
    return fdocs

def to_int(sent):
    start = int(sent[0: sent.find('-')])
    end = int(sent[sent.find('-') + 1: ])
    return [start, end]

def merge(fdocs):
    fdocs.sort(key= lambda x: x[0])
    result = []
    for fdoc in fdocs:
        if len(result) == 0 or fdoc[0] > result[-1][1] + 1:
            result.append(fdoc)
        else:
            result[-1][1] = fdoc[1]
    return result

def to_df(fdocs, id):
    result = []
    for fdoc in fdocs:
        new_row = {
            'id': id,
            'name': doccano_client.get_project_detail(id)['name'],
            'finish': str(fdoc[0]) + '-' + str(fdoc[1]),
            'link': pre_link + str(id)
        }
        result.append(new_row)
    return pd.DataFrame(result)

def extract_request(request):
    project_id = request.args.get('projectId')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))
    if start < 0:
        raise Exception('Start must be greater than 0.')

    if start >= end:
        raise Exception('Start must be less than or equal to End.')

    if 'detail' in doccano_client.get_project_detail(project_id):
        raise Exception('Project is not found.')

    total = doccano_client.get_project_statistics(project_id)['total']
    if end > total:
        raise Exception(
            f'End exceeds the total number of mention in this project. Maximum is {total}.')

    return project_id, start, end

if __name__=="__main__":
    project = {
        'id': 836,
        'start': 300,
        'end': 505
    }
    add_finish(project)
