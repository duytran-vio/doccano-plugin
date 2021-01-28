import json
from doccano_api_client import DoccanoClient

def to_sequence(documents):
    # if 'start_offset' not in documents[0]['annotations'][0].keys(): return documents
    new_doc = []
    for doc in documents:
        text = doc['text']
        for annotation in doc['annotations']:
            new_sequence = {'text': '', 'annotations': []}
            start = annotation['start_offset']
            end = annotation['end_offset']
            str = text[start: end]
            new_sequence['text'] = str
            new_sequence['annotations'].append(annotation)
            new_doc.append(new_sequence)
    return new_doc

def map_labels(labels_map, documents):
    for doc in documents:
        doc['labels'] = []
        for annotation in doc['annotations']:
            doc['labels'].append(labels_map[annotation['label']])
        del doc['annotations']


def build_label_map(doccano_client: DoccanoClient, project_id):
    labels = doccano_client.get_label_list(project_id)
    labels_map = {}
    for label in labels:
        labels_map[label['id']] = label['text']
    return labels_map


def get_all_documents(doccano_client: DoccanoClient, project_id):
    documents = doccano_client.get_document_list(project_id, {
        'limit': [doccano_client.get_project_statistics(project_id)['total']],
        'offset': [0],
    })['results']
    for doc in documents:
        doc['meta'] = json.loads(doc['meta'])
    return documents


def get_user_by_name(doccano_client: DoccanoClient, username):
    users = doccano_client.get_user_list()
    for user in users:
        if user['username'] == username:
            return user
    return None
