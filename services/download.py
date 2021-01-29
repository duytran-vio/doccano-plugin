import os
import jsonlines
import pandas as pd
from doccano_api_client import DoccanoClient

from .common import build_label_map, map_labels, get_all_documents, to_sequence


doccano_client: DoccanoClient = None
download_dir = 'download'


def handle_request(request, client: DoccanoClient):
    global doccano_client
    doccano_client = client
    project_id = extract_request(request)
    # project_detail = doccano_client.get_project_detail(project_id)
    truth_documents = get_all_documents(doccano_client, project_id)
    # if project_detail['project_type'] == 'SequenceLabeling':
    #     truth_documents = to_sequence(truth_documents)
    labels_map = build_label_map(doccano_client, project_id)
    # map_labels(labels_map, truth_documents)

    # for document in truth_documents:
    #     document['labels'] = ','.join(document['labels'])

    sequence_label_table = to_label_table(truth_documents, labels_map)

    file_name = f'{project_id}.xlsx'
    file_path = os.path.join(download_dir, file_name)
    pd.DataFrame(sequence_label_table).to_excel(file_path, index=False, engine='xlsxwriter')
    return file_name

def to_label_table(documents, labels_map):
    new_documents = []
    max_intent = 5
    for doc in documents:
        for annotation in doc['annotations']:
            new_doc = {}
            new_doc['text'] = doc['text'][max_intent:]
            start = annotation['start_offset']
            end = annotation['end_offset']
            sequence = doc['text'][start:end]
            if sequence[0] == '@' and end < max_intent: 
                sequence = doc['text'][max_intent:]
            sequence_label_id = annotation['label']
            new_doc[labels_map[sequence_label_id]] = sequence
            new_documents.append(new_doc)
    return new_documents

def extract_request(request):
    project_id = request.args.get('projectId')

    if 'detail' in doccano_client.get_project_detail(project_id):
        raise Exception('Project is not found.')

    return project_id
