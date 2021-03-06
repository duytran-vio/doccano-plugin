from os import path
import json
import random
import jsonlines
import pandas as pd
import re

from .common import build_label_map, map_labels, to_input_sequence, replace_double_quotes, get_documents

doccano_client = None
ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
DATA_PATH = path.join(ROOT, 'doccano_project_data')
DATA_SAMPLE_PATH = path.join(DATA_PATH, 'sample')


def handle_request(request, client):
    global doccano_client
    doccano_client = client
    project_id, start, end, sample_size, new_project_name, username = extract_request(request)
    labels_map = build_label_map(doccano_client, project_id)
    indexes, documents = sample_documents(project_id, start, end, sample_size, labels_map)
    file_name = f'{project_id}-{abs(hash(" ".join(map(str, indexes))))}.jsonl'
    file_path = f'{DATA_SAMPLE_PATH}/{file_name}'

    response = doccano_client.create_project(
        name=f'{new_project_name}-{start + 1}-{end}',
        description=file_name,
        project_type='SequenceLabeling',
        resourcetype= "SequenceLabelingProject",
        collaborative_annotation=True,
    )
    new_project_id = response['id']
    create_labels(new_project_id)
    assign_user(new_project_id, username)
    
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(documents)

    try:
        doccano_client.post_doc_upload(new_project_id, 'json', file_name, DATA_SAMPLE_PATH)
    except json.JSONDecodeError:
        pass
    return new_project_id

def extract_request(request):
    body = request.get_json(force=True)
    project_id = body['projectId']
    start = body['start']
    end = body['end']
    sample_size = body['sampleSize']
    new_project_name = body['newProjectName']
    username = body['username']

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

    users = doccano_client.get_user_list()
    if not any([user['username'] == username for user in users]):
        raise Exception('User is not found.')
    return project_id, start - 1, end, sample_size, new_project_name, username


def sample_documents(project_id, start, end, sample_size, labels_map):
    orig_docs = get_documents(doccano_client, project_id, start, end)
    cus_docs = []
    for doc in orig_docs:
        sent = doc['text']
        sent = re.sub('@+','', sent)
        sent = re.sub('^\s+','', sent)
        if re.search('^khách', sent.lower()) is not None:
            cus_docs.append(doc)
    indexes = sorted(random.sample(range(0, len(cus_docs)), min(sample_size, len(cus_docs))))
    documents = []
    for index in indexes:
        documents.append(cus_docs[index])
    # labels_map = build_label_map(doccano_client, project_id)
    documents = to_input_sequence(documents, labels_map)
    return indexes, documents


def create_labels(project_id):
    labels = json.load(open('./category.labels.json', encoding='utf-8'))
    for label in labels:
        doccano_client.create_label(
            project_id=project_id,
            text=label['text'],
            prefix_key=label['prefix_key'],
            suffix_key=label['suffix_key'],
            background_color=label['background_color'],
            text_color=label['text_color'],
        )


def assign_user(project_id, username):
    roles = doccano_client.get_rolemapping_list(project_id)
    for role in roles:
        if role['username'] == username:
            doccano_client.delete(f'v1/projects/{project_id}/roles/{role["id"]}')
            break
    users = doccano_client.get_user_list()
    for user in users:
        if user['username'] == username:
            doccano_client.post(f'v1/projects/{project_id}/roles', data={'role': 1, 'user': user['id']})

if __name__ == "__main__":
    print(DATA_PATH)