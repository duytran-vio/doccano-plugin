from os import path
import json
import jsonlines
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import classification_report

from .common import build_label_map, map_labels, get_all_documents

doccano_client = None
labels = list(map(lambda x: x['text'], json.load(
    open('./category.labels.json', encoding='utf-8'))))

intent_boundary = 6 # max_intent + 1

ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
DATA_PATH = path.join(ROOT, 'doccano_project_data')
DATA_SAMPLE_PATH = path.join(DATA_PATH, 'sample')
DATA_DOWNLOAD_PATH = path.join(DATA_PATH, 'download')

def handle_request(request, client):
    global doccano_client
    doccano_client = client

    project_id = extract_request_2(request)
    sample_file_name = doccano_client.get_project_detail(project_id)["description"]
    initial_project_id = int(sample_file_name[:sample_file_name.find('-')])

    json_list = list(open(f'{DATA_SAMPLE_PATH}/{sample_file_name}', 'r', encoding='utf-8'))
    truth_documents = []
    for json_str in json_list:
        result = json.loads(json_str)
        truth_documents.append(result)

    predict_documents = get_all_documents(doccano_client, project_id)
    predict_labels_map = build_label_map(doccano_client, project_id)

    summary = []
    for i in range(len(predict_documents)):
        truth_intents, truth_sequences = get_sequence_truth_doc(truth_documents[i])
        predict_intents, predict_sequences = get_sequence(predict_documents[i], predict_labels_map)

        truth_intents, predict_intents = preprocess_intents(truth_intents, predict_intents, predict_labels_map)
        text = truth_documents[i]['text']
        if truth_intents != predict_intents:
            intent_summary = {
                'text': text[intent_boundary:], 
                'Wrong Label': truth_intents, 
                'True Label': predict_intents
            }
            summary.append(intent_summary)

        update_sqs = compair(truth_sequences, predict_sequences, predict_labels_map)
        for sequence in update_sqs:
            start = sequence['start_offset']
            end = sequence['end_offset']
            entity_summary = {
                'text': text[start:end], 
                'Wrong Label': sequence['Wrong Label'], 
                'True Label': sequence['True Label']
            }
            summary.append(entity_summary)
    file_name = f'Tesing_{initial_project_id}_Sample_{project_id}.xlsx'
    file_path = path.join(DATA_DOWNLOAD_PATH, file_name)
    pd.DataFrame(summary).to_excel(file_path, index=False, engine='xlsxwriter')

    return file_name

def preprocess_intents(truth_intents, predict_intents, predict_labels_map):
    if truth_intents:
        truth_intents = ','.join(truth_intents)
    else:
        truth_intents = ''
    
    if predict_intents:
        predict_intents = ','.join(predict_intents)
    else: 
        predict_intents = ''
    return truth_intents, predict_intents

def compair(truth_sequences, predict_sequences, predict_labels_map):
    update_sqs = []
    is_add = [True for i in range(len(predict_sequences))]
    is_del = [True for i in range(len(truth_sequences))]
    for i in range(len(predict_sequences)):
        for j in range(len(truth_sequences)):
            predict_sq = predict_sequences[i]
            truth_sq = truth_sequences[j]
            predict_label = predict_labels_map[predict_sq['label']]
            truth_label = truth_sq['label']
            if predict_sq['start_offset'] == truth_sq['start_offset'] and predict_sq['end_offset'] == truth_sq['end_offset']:
                if predict_label != truth_label:
                    update_sqs.append(
                        {
                            'start_offset': predict_sq['start_offset'],
                            'end_offset': predict_sq['end_offset'],
                            'True Label': predict_label, 
                            'Wrong Label': truth_label
                        })
                is_add[i] = False
                is_del[j] = False
    for i in range(len(is_del)):
        if is_del[i]:
            truth_sq = truth_sequences[i]
            update_sqs.append(
                {
                    'start_offset': truth_sq['start_offset'],
                    'end_offset': truth_sq['end_offset'],
                    'True Label': '', 
                    'Wrong Label': truth_sq['label']

                })
    for i in range(len(is_add)):
        if is_add[i]:
            predict_sq = predict_sequences[i]
            update_sqs.append(
                {
                    'start_offset': predict_sq['start_offset'],
                    'end_offset': predict_sq['end_offset'],
                    'True Label': predict_labels_map[predict_sq['label']], 
                    'Wrong Label': ''
                })
    return update_sqs

def get_sequence(document, labels_map):
    list_sequence = []
    intents = []
    have_intent = document['text'][0] == '@'
    for annotation in document['annotations']:
        sequence = {}
        sequence['start_offset'] = annotation['start_offset']
        sequence['end_offset'] = annotation['end_offset']
        sequence['label'] = annotation['label']
        if have_intent and sequence['start_offset'] <= intent_boundary:
            intents.append(sequence['label'])
        else:
            list_sequence.append(sequence)
    for i in range(len(intents)):
        intents[i] = labels_map[intents[i]]
    intents.sort()
    list_sequence = sorted(list_sequence, key = lambda i: (i['start_offset']))
    return intents, list_sequence

def get_sequence_truth_doc(document):
    list_sequence = []
    intents = []
    have_intent = document['text'][0] == '@'
    for label in document['labels']:
        sequence = {}
        sequence['start_offset'] = label[0]
        sequence['end_offset'] = label[1]
        sequence['label'] = label[2]
        if have_intent and sequence['start_offset'] <= intent_boundary:
            intents.append(sequence['label'])
        else:
            list_sequence.append(sequence)
    intents.sort()
    list_sequence = sorted(list_sequence, key= lambda i: (i['start_offset']))
    return intents, list_sequence

def extract_request_2(request):
    project_id = request.args.get('projectId')

    if 'detail' in doccano_client.get_project_detail(project_id):
        raise Exception('Project is not found.')

    return project_id
