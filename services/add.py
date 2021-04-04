import json
import jsonlines
import pandas as pd
from werkzeug.utils import secure_filename
import os
from .classifier import classifier as labeling_docs
from .common import to_input_sequence, get_all_documents, build_label_map


doccano_client = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'doccano_project_data')
DATA_CREATE_PATH = os.path.join(DATA_PATH, 'create')
DATA_UPLOAD_PATH = os.path.join(DATA_PATH, 'upload')

def handle_request(request, client):
    global doccano_client
    doccano_client = client
    project_id, documents_file = extract_request(request)

    df_documents = labeling_docs(data_file_path=documents_file)
    new_documents = df_documents.to_dict('records')
    current_total = doccano_client.get_project_statistics(project_id)['total']

    file_name = f'{project_id}_{current_total + 1}-{current_total + len(new_documents)}_docs'
    file_path = f'{DATA_CREATE_PATH}/{file_name}'
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(new_documents)

    try:
        doccano_client.post_doc_upload(project_id, 'json', file_name, DATA_CREATE_PATH)
    except json.JSONDecodeError:
        pass

    return project_id

def intent_jsonl_form(documents):
    for i in range(len(documents)):
        n_intents = 0
        ls_intents = []
        for label in documents[i]['labels']:
            ls_intents.append([n_intents, n_intents + 1, label])
            n_intents = n_intents + 1
        documents[i]['labels'] = ls_intents
    return documents

def extract_request(request):
    project_id = request.form['projectId']
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(DATA_UPLOAD_PATH, filename)
    file.save(path)
    return project_id, path

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
