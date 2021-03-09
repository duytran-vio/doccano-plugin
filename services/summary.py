import os
import jsonlines
import pandas as pd
from doccano_api_client import DoccanoClient

from .common import build_label_map, map_labels, get_all_documents


doccano_client: DoccanoClient = None
download_dir = 'download'
intent_boundary = 6 # max_intent + 1
list_label = [
    'Hello', 'Done', 'Connect', 'Order',
    'Changing', 'Return', 'Other', 'Inform',
    'Request', 'feedback','Shop_Hello', 'Shop_Done', 'Shop_not-found',
    'Shop_inform', 'Shop_request', 'Shop_connect', 'Shop_confirm',
    'Shop_Reject', 'Shop_other', 'ID_product', 'size_product',
    'color_product', 'material_product', 'cost_product',
    'amount_product', 'Id member', 'phone', 'addr member', 
    'shiping fee', 'size customer', 
    'height customer', 'weight customer',
    'addr store'
]

def handle_request(request, client: DoccanoClient):
    global doccano_client
    doccano_client = client
    project_id, start, end = extract_request(request)
    truth_documents = get_all_documents(doccano_client, project_id)
    truth_documents = truth_documents[start:end]
    labels_map = build_label_map(doccano_client, project_id)
    sequence_label_table = to_label_table(truth_documents, labels_map)

    #count labeled doc
    n_is_label = 0
    for doc in truth_documents:
        if len(doc['annotations']) != 0:
            n_is_label = n_is_label + 1
    
    #count number of sequence per label
    summary = get_summary(sequence_label_table, list_label)

    #append number of labeled doc to summary
    summary = summary.append({'label':'Labeled docs', 'count': n_is_label}, ignore_index = True)

    summary_file_name = f'{project_id}-{start}-{end}_summary.xlsx'
    summary_path = os.path.join(download_dir, summary_file_name)
    summary.to_excel(summary_path, index=False, engine='xlsxwriter')
    return summary_file_name

def get_summary(sequence_label_table, list_label):
    cnt = {}
    for label in list_label:
        cnt[label] = 0

    for k in range(len(sequence_label_table)):
        for label in list_label:
            if label in sequence_label_table[k].keys():
                cnt[label] = cnt[label] + len(sequence_label_table[k][label])

    summary = pd.DataFrame([label for label in list_label], columns = ['label'])
    summary['count'] = [cnt[label] for label in list_label]
    return summary

def to_label_table(documents, labels_map):
    new_documents = []
    for doc in documents:
        text = doc['text']
        if doc['text'][0] == '@':
            text = doc['text'][intent_boundary:]
        new_doc = {'id' : doc['id'], 'text': text}
        new_doc.update((labels_map[k], list()) for k in labels_map)
        for annotation in doc['annotations']:
            start = annotation['start_offset']
            end = annotation['end_offset']
            sequence = doc['text'][start:end]
            if sequence[0] == '@' and end <= intent_boundary: 
                sequence = doc['text'][intent_boundary:]
            sequence_label_id = annotation['label']
            label = labels_map[sequence_label_id]
            new_doc[label].append(sequence)
        new_documents.append(new_doc)    
    return new_documents

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

    return project_id, start - 1, end
