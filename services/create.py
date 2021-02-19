import json
import jsonlines
import pandas as pd
from werkzeug.utils import secure_filename
import os
from .classifier import classifier as labeling_docs


doccano_client = None

def handle_request(request, client):
    global doccano_client
    doccano_client = client
    project_name, project_des, documents_file = extract_request(request)
    df_documents = labeling_docs(data_file_path=documents_file)
    documents = df_documents.to_dict('records')
    documents = intent_jsonl_form(documents)

    response = doccano_client.create_project(
        name=project_name,
        description=project_des,
        project_type='SequenceLabeling',
        resourcetype= "SequenceLabelingProject",
        collaborative_annotation=True,
    )
    new_project_id = response['id']
    create_labels(new_project_id)

    file_name = f'{new_project_id}_docs'
    file_path = f'tmp/{file_name}'
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(documents)

    try:
        doccano_client.post_doc_upload(new_project_id, 'json', file_name, 'tmp')
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
    path = os.path.join('upload', filename)
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
