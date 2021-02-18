import os
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

download_dir = 'download'
intent_boundary = 6 # max_intent + 1

def handle_request(request, client):
    global doccano_client
    doccano_client = client

    project_id = extract_request_2(request)
    # initial_project_id = int(doccano_client.get_project_detail(project_id)["description"][0:3])
    description = doccano_client.get_project_detail(project_id)["description"]
    initial_project_id = int(description[:description.find('-')])
            
    predict_documents = get_all_documents(doccano_client, project_id)
    truth_documents = get_all_documents(doccano_client, initial_project_id)

    evaluate_documents = []
    k = 0
    for doc in truth_documents:
        if doc['text'] == predict_documents[k]['text']:
            k = k + 1
            evaluate_documents.append(doc)
            if k == len(predict_documents):
                break
    truth_documents = evaluate_documents
 
    predict_labels_map = build_label_map(doccano_client, project_id)
    truth_labels_map = build_label_map(doccano_client, initial_project_id)

    summary = []
    for i in range(len(predict_documents)):
        truth_intents, truth_sequences = get_sequence(truth_documents[i])
        predict_intents, predict_sequences = get_sequence(predict_documents[i])

        truth_intents, predict_intents = preprocess_intents(truth_intents, predict_intents, truth_labels_map, predict_labels_map)
        text = truth_documents[i]['text']
        if truth_intents != predict_intents:
            intent_summary = {
                'text': text[intent_boundary:], 
                'Wrong Label': truth_intents, 
                'True Label': predict_intents
            }
            summary.append(intent_summary)
        update_sqs = compair(truth_sequences, predict_sequences, truth_labels_map, predict_labels_map)
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
    file_path = os.path.join(download_dir, file_name)
    pd.DataFrame(summary).to_excel(file_path, index=False, engine='xlsxwriter')

    return file_name

def preprocess_intents(truth_intents, predict_intents, truth_labels_map, predict_labels_map):
    for i in range(len(truth_intents)):
        truth_intents[i] = truth_labels_map[truth_intents[i]]
    for i in range(len(predict_intents)):
        predict_intents[i] = predict_labels_map[predict_intents[i]]
    if truth_intents:
        truth_intents = ','.join(truth_intents)
    else:
        truth_intents = ''
    
    if predict_intents:
        predict_intents = ','.join(predict_intents)
    else: 
        predict_intents = ''
    return truth_intents, predict_intents

def compair(truth_sequences, predict_sequences, truth_labels_map, predict_labels_map):
    update_sqs = []
    is_add = [True for i in range(len(predict_sequences))]
    is_del = [True for i in range(len(truth_sequences))]
    for i in range(len(predict_sequences)):
        for j in range(len(truth_sequences)):
            predict_sq = predict_sequences[i]
            truth_sq = truth_sequences[j]
            predict_label = predict_labels_map[predict_sq['label']]
            truth_label = truth_labels_map[truth_sq['label']]
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
                    'Wrong Label': truth_labels_map[truth_sq['label']]

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

def get_sequence(document):
    list_sequence = []
    intents = []
    for annotation in document['annotations']:
        sequence = {}
        sequence['start_offset'] = annotation['start_offset']
        sequence['end_offset'] = annotation['end_offset']
        sequence['label'] = annotation['label']
        if document['text'][0] == '@' and sequence['end_offset'] <= intent_boundary:
            intents.append(sequence['label'])
        else:
            list_sequence.append(sequence)
    intents.sort()
    list_sequence = sorted(list_sequence, key = lambda i: (i['start_offset']))
    return intents, list_sequence

def extract_request_2(request):
    project_id = request.args.get('projectId')

    if 'detail' in doccano_client.get_project_detail(project_id):
        raise Exception('Project is not found.')

    return project_id


def extract_prediction(truth_documents, predict_documents):
    y_true = list(map(lambda doc: doc['labels'], truth_documents))
    ids = list(map(lambda doc: doc['meta']['id'], truth_documents))
    predict_document_dict = {}
    for doc in predict_documents:
        predict_document_dict[doc['meta']['id']] = doc
    y_pred = [predict_document_dict[i]['labels'] for i in ids]
    return y_true, y_pred


def calculate_macro_f1_score(true_labels: List[List[str]], pred_labels: List[List[str]]):
    y_true, y_pred = [], []
    for true_label, pred_label in zip(true_labels, pred_labels):
        y_true_bin = [0] * len(labels)
        y_pred_bin = [0] * len(labels)
        for i, label in enumerate(labels):
            if any([tl == label for tl in true_label]):
                y_true_bin[i] = 1
            if any([pl == label for pl in pred_label]):
                y_pred_bin[i] = 1
        y_true.append(y_true_bin)
        y_pred.append(y_pred_bin)

    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        target_names=labels,
    )
    valid_f1_scores = []
    for label in labels:
        if report[label]['support'] > 0:
            valid_f1_scores.append(report[label]['f1-score'])
    return {
        'f1-score': round(np.mean(valid_f1_scores), 2),
        'categories': [{
            'category': label,
            'precision': round(report[label]['precision'], 2),
            'recall': round(report[label]['recall'], 2),
            'f1-score': round(report[label]['f1-score'], 2),
            'support': report[label]['support'],
        } for label in labels],
    }
