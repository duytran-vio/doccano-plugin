import json
import jsonlines
import pandas as pd
from werkzeug.utils import secure_filename
import os

doccano_client = None
file_name = f'data/data.txt'

def handle_request(request, client):
    global doccano_client
    doccano_client = client
    project_name, project_des, documents_file = extract_request(request)
    fi = open(documents_file, encoding='utf-8')
    documents = fi.readlines()

    response = doccano_client.create_project(
        name=project_name,
        description=project_des,
        project_type='SequenceLabeling',
        resourcetype= "SequenceLabelingProject",
        collaborative_annotation=True,
    )
    new_project_id = response['id']
    create_labels(new_project_id)

    return new_project_id

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
