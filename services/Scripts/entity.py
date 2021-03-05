from vncorenlp import VnCoreNLP
import os
import re
import pandas as pd

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# BASEDIR = 'D:\\GitHub\\VnCoreNLP'
vncorenlp_file = os.path.join(BASEDIR,os.path.join('VnCoreNLP','VnCoreNLP-1.1.1.jar'))

list_entity_using_regex = ['phone', 'weight customer', 'shiping fee', 'height customer', 'size_product']
pattern_list = {
    'phone': [
        r'[0-9]{4}\.*[0-9]{3}\.*[0-9]{2,4}', 
        r'[0-9|.]*xx+'
    ],
    'weight customer': [
        r'\d+\s*(kg|ky|ký|ki+|kí+)'
    ],
    'shiping fee':[
        r'((miễn|free|\d+k*)\s*)*ship(\s*\d+k*)*', 
        r'((\d+k*\s*)*(phí|giá|gia|phi|tiền|tien)\s*)ship(\s*\d+k*)*'
    ],
    'height customer':[
        r'((\dm|m)\d+|\d+cm)'
    ],
    'size_product':[
        r'(size|sai|sz)\s(\d*[smlx]*[SMLX]*)'
    ]
}

def label_entity(sentences):
    sents_entity = [[] for i in range(len(sentences))]
    for entity in list_entity_using_regex:
        for i in range(len(sentences)):
            sent = sentences['text'][i].lower()
            list_entity_sq = get_entity_sq_from_list_pt(pattern_list[entity], sent, entity)
            if len(list_entity_sq) > 0:
                sents_entity[i].extend(list_entity_sq)
            
    ## Use ner to get Id member entity
    # vncorenlp = VnCoreNLP(vncorenlp_file)
    # for i in range(len(sentences)):
    #     sent = sentences['text'][i]
    #     # print(sent)
    #     list_Id_member_sq = infer_Id_member(sent,vncorenlp)
    #     # print(list_Id_member_sq)
    #     if len(list_Id_member_sq) > 0:
    #         sents_entity[i].extend(list_Id_member_sq)

    # vncorenlp.close()

    return sents_entity

def preprocess_ner(sentence):
    sentence = re.sub(r'_',' ', sentence)
    sentence = re.sub(r'[!@#$%^&*<>?,.:;]+', '', sentence)
    ### remove multiple white spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    ### remove start and end white spaces
    sentence = re.sub(r'^\s+', '', sentence) 
    sentence = re.sub(r'\s+$', '', sentence)
    return sentence

def infer_Id_member(sentences, vncorenlp):
    sents_ner = vncorenlp.ner(sentences)
    # print(sents_ner)
    list_id_member = []
    for sent_ner in sents_ner:
        for ents in sent_ner:
            if ents[1][2:] == 'PER':
                list_id_member.append(ents[0])
    # print(list_id_member)

    k = 0
    list_sq = []
    for i in range(len(list_id_member)):
        text = list_id_member[i]
        text = preprocess_ner(text)
        result = re.search(text, sentences[k:])
        if result is None: 
            continue
        idx = result.span()
        sq = [idx[0] + k, idx[1] + k, 'Id member']
        list_sq.append(sq)
        k = idx[1]

    # join continuous sequence
    res = []
    for sequence in list_sq:
        if len(res) == 0:
            res.append(sequence)
            continue
        start = res[-1][1]
        end = sequence[0]
        if re.search(r'^\s+$', sentences[start:end]) is not None:
            res[-1][1] = sequence[1]
        else:
            res.append(sequence)
    return res

def findall_index(pattern, sent, entity):
    list_sub = []
    k = 0
    while 1 > 0:
        p = re.search(pattern, sent[k: ])
        if p is None: break
        idx = p.span()
        list_sub.append([idx[0], idx[1], entity])
        k = idx[1]
    return list_sub

def get_entity_sq_from_list_pt(list_pattern, sent, entity):
    list_entity_sq = []
    for pattern in list_pattern:
        list_index_pt = findall_index(pattern, sent, entity)
        if len(list_index_pt) > 0:
            list_entity_sq.extend(list_index_pt)
    return list_entity_sq

if __name__ == "__main__":
    print(vncorenlp_file)
    vncorenlp = VnCoreNLP(vncorenlp_file)
    vncorenlp.close()
