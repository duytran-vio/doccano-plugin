import os
import jsonlines
import pandas as pd
from doccano_api_client import DoccanoClient

from .common import build_label_map, map_labels, get_all_documents


doccano_client: DoccanoClient = None
download_dir = 'download'
intent_boundary = 6 # max_intent + 1

def handle_request(request, client: DoccanoClient):
    global doccano_client
    doccano_client = client
    project_id = extract_request(request)
    truth_documents = get_all_documents(doccano_client, project_id)
    labels_map = build_label_map(doccano_client, project_id)
    sequence_label_table = to_label_table(truth_documents, labels_map)

    # for doc in sequence_label_table:
    cnt = {}
    for i in labels_map: 
        cnt[i] = 0

    for k in range(len(sequence_label_table)):
        for i in labels_map:
            cnt[i] = cnt[i] + len(sequence_label_table[k][labels_map[i]])
            sequence_label_table[k][labels_map[i]] = '|/|'.join(sequence_label_table[k][labels_map[i]])

    summary = pd.DataFrame([labels_map[i] for i in labels_map], columns = ['label'])
    summary['count'] = [cnt[i] for i in cnt]

    file_name = f'{project_id}.xlsx'
    file_path = os.path.join(download_dir, file_name)
    summary_file_name = f'{project_id}_summary.xlsx'
    summary_path = os.path.join(download_dir, summary_file_name)
    pd.DataFrame(sequence_label_table).to_excel(file_path, index=False, engine='xlsxwriter')
    summary.to_excel(summary_path, index=False, engine='xlsxwriter')
    return file_name, summary_file_name

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

    if 'detail' in doccano_client.get_project_detail(project_id):
        raise Exception('Project is not found.')

    return project_id
