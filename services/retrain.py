import os
import csv
import pandas as pd
from doccano_api_client import DoccanoClient

doccano_client: DoccanoClient = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'doccano_project_data')
DATA_SAMPLE_PATH = os.path.join(DATA_PATH, 'sample')
DATA_DOWNLOAD_PATH = os.path.join(DATA_PATH, 'download')
RETRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'retrain')

def handle_request(request, client: DoccanoClient):
    global doccano_client
    doccano_client = client
    project_id, start, end = extract_request(request)
    RETRAIN_PROJECT_FILE = os.path.join(RETRAIN_PATH, 'retrain_project.csv')
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
