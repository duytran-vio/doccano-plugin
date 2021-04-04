import json
import jsonlines
import pandas as pd
from werkzeug.utils import secure_filename
import os
from .classifier import classifier as labeling_docs


doccano_client = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'doccano_project_data')
DATA_CREATE_PATH = os.path.join(DATA_PATH, 'create')
DATA_UPLOAD_PATH = os.path.join(DATA_PATH, 'upload')

def handle_request(request, client):
    global doccano_client
    doccano_client = client
    project_name, project_des, documents_file = extract_request(request)
    df_documents = labeling_docs(data_file_path=documents_file)
    documents = df_documents.to_dict('records')

    response = doccano_client.create_project(
        name=project_name,
        description=project_des,
        project_type='SequenceLabeling',
        resourcetype= "SequenceLabelingProject",
        collaborative_annotation=True,
    )
    new_project_id = response['id']
    create_labels(new_project_id)

    file_name = f'{new_project_id}_1-{len(documents)}_docs'
    file_path = f'{DATA_CREATE_PATH}/{file_name}'
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(documents)

    try:
        doccano_client.post_doc_upload(new_project_id, 'json', file_name, DATA_CREATE_PATH)
    except json.JSONDecodeError:
        pass

    return new_project_id

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
    project_name = request.form['projectName']
    project_des = request.form['projectDes']
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(DATA_UPLOAD_PATH, filename)
    file.save(path)
    return project_name, project_des, path

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
