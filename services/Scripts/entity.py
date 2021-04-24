from vncorenlp import VnCoreNLP
from os import path
import re
import pandas as pd
from .address import address_entity
import numpy as np
import time
import json

BASEDIR = path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
# BASEDIR = 'D:\\GitHub\\VnCoreNLP'
vncorenlp_file = path.join(BASEDIR,path.join('VnCoreNLP','VnCoreNLP-1.1.1.jar'))

MODELS_PATH = path.join(
    path.dirname(path.dirname(path.abspath(__file__))),
    'models'
)

### COMMON
product_pt = '[á|a]o|qu[a|ầ]n|v[a|á]y|đầm|dam|t[ú|u]i|n[ó|o]n|m[u|ũ]|kho[a|á]c'
df_amount_suf = pd.read_csv(path.join(MODELS_PATH, 'amount_suf.csv'), header = None)
amount_suf = df_amount_suf[0].tolist()
pt_amount_suf = '|'.join(amount_suf)
###------------------------------------------

### COLOR_PRODUCT
df_colors = pd.read_csv(path.join(MODELS_PATH, 'colors.csv'), header=None)
df_colors_2 = pd.read_csv(path.join(MODELS_PATH, 'colors_2.csv'), header = None)
colors = df_colors[0].tolist()
colors_2 = df_colors_2[0].tolist()

pt_color_pref = r'(m[à|a]u|{}|{})'.format(product_pt, pt_amount_suf)
pt_color_1 = r'\b('+ '|'.join(colors) + r')(\s(đậm|dam|nhạt|nhat))*\b'
pt_color_2 = r'\b({})\s*('.format(pt_color_pref) + '|'.join(colors_2) + r')(\s(đậm|dam|nhạt|nhat))*\b'

pt_color = r'{}|{}'.format(pt_color_1, pt_color_2)
###------------------------------------------
### COST_PRODUCT
dong_pt = r'đồng|dong|đ|dog|VND|VNĐ'
cost_pt = r'\b\d+\s*(k(\s{0:}|\d*)*|tr((iệ|ie)u)*(\s{0:}|\d*)*|ng[a|à]n(\s{0:}|\d*)*|t[ỉiỷy](\s{0:}|\d*)*|{0:})\b'.format(dong_pt)
cost_pt_sum = '{}'.format(cost_pt)
###------------------------------------------

### AMOUNT_PRODUCT

amount_pt = r'(\d+-)*\d+\s*(' + '|'.join(amount_suf) + r')((\s({})*)|(?=[^a-z]|$))'.format(product_pt) 
amount_pt_2 = r'(\d+-)*\d+\s*({})'.format(product_pt)
amount_pt_sum = r'{0:}|{1:}'.format(amount_pt, amount_pt_2)
###------------------------------------------

### MATERIAL_PRODUCT
df_material = pd.read_csv(path.join(MODELS_PATH, 'material.csv'), header = None)
material = df_material[0].tolist()
pt_material = r'\b((ch[a|ấ]t(\sli[e|ệ]u)*|lo[a|ạ]i)\s)*(' + '|'.join(material) + r')(\sc[u|ứ]ng|\sm[e|ề]m)*\b'
###------------------------------------------

### SIZE
size_pref = r'size|sai|sz|c[a|á]i'
size_main = r'\d*(x*s|m|x*l|a|nhỏ|lớn|nho|lon)'
pt_size_1 = r'\b({}|{})\s{}\b'.format(size_pref, product_pt, size_main)
pt_size_2 = r'\b({})\s\d+\b'.format(size_pref)
pt_size_3 = r'\b(\d*(x*s|x*l))\b'
pt_size = r'{}|{}|{}'.format(pt_size_1, pt_size_2, pt_size_3)
###------------------------------------------

### list of pattern
list_entity_using_regex = ['phone', 'weight customer', 'height customer', 
                            'size', 'color_product','cost_product', 'shiping fee',
                            'amount_product', 'material_product', 'ID_product'
                            ]
pattern_list = {
    'phone': [
        r'\b[0-9]{4}\.*[0-9]{3}\.*[0-9]{3,4}\b'
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
    'size':[
        pt_size
    ],
    'color_product':[
        pt_color
    ],
    'cost_product':[
        cost_pt_sum
    ],
    'amount_product':[
        amount_pt_sum
    ],
    'material_product':[
        pt_material
    ]
}

file_char = open(path.join(MODELS_PATH, 'list_char.json'), 'r')
file_ID_product = open(path.join(MODELS_PATH, 'ID_product.json'), 'r')
list_char = json.load(file_char)
trie = json.load(file_ID_product)

def label_entity(sentences, address_inp):
    '''
    Argument: 
        sentences: one string need to label

    return:
        sents_entity: list [start_offset, end_offset, label_name]
    '''
    sents_entity = [[] for i in range(len(sentences))]
    ner_entity = [[] for i in range(len(sentences))]

    ## Use ner to get Id member entity
    '''
    Uncomment below to get Id member entity
    '''
    # with VnCoreNLP(address='http://127.0.0.1', port=9000) as vncorenlp:
    #     for i in range(len(sentences)):
    #         sent = sentences[i]
    #         # print(sent)
    #         list_Id_member_sq = infer_Id_member(sent,vncorenlp)
    #         list_Id_member_sq = reduce_label(list_Id_member_sq, sent.find(':'))
    #         # print(list_Id_member_sq)
    #         ner_entity[i] = list_Id_member_sq

    '''
    Using regex
    '''
    for i in range(len(sentences)):
        sent = sentences[i].lower()
        result = []
        for entity in list_entity_using_regex:
            if entity == 'ID_product':
                list_entity_sq = infer_ID_product(trie, sent)
            elif entity == 'cost_product' or entity == 'shiping fee':
                list_sq = label_full_string(sent)
                list_entity_sq = [e for e in list_sq if e[2] == entity]
            else:
                list_entity_sq = get_entity_sq_from_list_pt(pattern_list[entity], sent, entity)
            list_entity_sq = join_continuous_sq(list_entity_sq, sent)
            list_entity_sq = reduce_label(list_entity_sq, sent.find(':'))
            if len(list_entity_sq) > 0:
                result.extend(list_entity_sq)

            ### ADDRESS
            sent = sentences[i].lower()
            sent = re.sub('-|,', ' ', sent)
            sent = re.sub('xxx', ' xx', sent)
            start, end, ent, score = address_entity(sent, address_inp)
            if score > 12:
                start, end = decode_start_end(sent, start, end)
                result.extend([(start, end, ent)])
        sents_entity[i] = remove_duplicate_entity(result, len(sent))

    ## Merge Id member to sents_entity
    '''
    Uncomment below to get Id member entity
    '''
    # for i in range(len(sentences)):
    #     sents_entity[i] = merge(ner_entity[i], sents_entity[i])

    return sents_entity

def decode_start_end(sent, start, end):
    from unidecode import unidecode
    sent = sent.lower()
    dec_sent = unidecode(sent)

    j = 0
    for i in range(len(dec_sent)):
        while unidecode(sent[j]) != dec_sent[i]:
            j+=1
        if i == start:
            dec_start = j
        elif i == end-1:
            dec_end = j
            break
        j+=1
    return dec_start, dec_end+1

def merge(ner_entity, list_sq):
    list_sq.sort(key= lambda i: i[1] - i[0], reverse=True)
    for i in range(len(list_sq)):
        check = True
        for j in range(len(ner_entity)):
            if list_sq[i][1] <= ner_entity[j][0] or list_sq[i][0] >= ner_entity[j][1]: 
                continue
            check = False
            break
        if check:
            ner_entity.append(list_sq[i])
    return ner_entity

def reduce_label(list_sq, boundary):
    if boundary < 0:
        res = list_sq
    else:
        res = [e for e in list_sq if e[1] > boundary]
    return res

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
    list_sq = join_continuous_sq(list_sq, sentences)
    return list_sq

def findall_index(pattern, sent, entity):
    list_sub = []
    k = 0
    while 1 > 0:
        p = re.search(pattern, sent[k: ])
        if p is None: break
        idx = p.span()
        list_sub.append([idx[0] + k, idx[1] + k, entity])
        k = k + idx[1]
    return list_sub

def get_entity_sq_from_list_pt(list_pattern, sent, entity):
    list_entity_sq = []
    for pattern in list_pattern:
        list_index_pt = findall_index(pattern, sent, entity)
        if len(list_index_pt) > 0:
            list_entity_sq.extend(list_index_pt)
    return list_entity_sq

def join_continuous_sq(list_sq, sentences):
    res = []
    for sequence in list_sq:
        if len(res) == 0:
            res.append(sequence)
            continue
        start = res[-1][1]
        end = sequence[0]
        if re.search(r'^\s+$', sentences[start:end]) is not None or start == end:
            res[-1][1] = sequence[1]
        else:
            res.append(sequence)
    return res

def return_pre_words(string, start, num_of_words):
  word = ""
  pre = start - 2
  pre_words = num_of_words
  while (pre >= 0 and pre_words > 0):
    word = string[pre] + word
    pre = pre - 1
    if (string[pre] == " "):
        pre_words = pre_words - 1
  
  return word

def return_next_words(string, end, num_of_words):
  word = ""
  next = end + 1
  next_words = num_of_words
  while (next < len(string) and next_words > 0):
    word = word + string[next]
    if (string[next] == " "):
        next_words = next_words - 1
    next = next + 1
  return word

def get_string_in_range(string, neighbor, start, end):
    new_str = return_pre_words(string, start, neighbor) + " " + string[start:end] + " " + return_next_words(string, end, neighbor)
    return new_str

def label_shipping_and_cost(string, neighbor=2):
    shipping = r'\b(ship|cod|phí|phi)\b'
    string = string.lower()
    find = re.search(cost_pt_sum, string)
    if (find):
        k = find.span()
        cost_sq = string[k[0]:k[1]]
        new_str = get_string_in_range(string, neighbor, k[0], k[1])
        re_value = re.search('\d+', cost_sq)
        value_idx = re_value.span()
        value = int(cost_sq[value_idx[0]: value_idx[1]])
        small_cost_60 = value <= 60 and re.search('tr((iệ|ie)u)*|t[ỉiỷy]', cost_sq) is None
        small_cost_30 = value <= 30 and re.search('tr((iệ|ie)u)*|t[ỉiỷy]', cost_sq) is None

        if re.search('free\s*ship', new_str) is None and ((small_cost_60 and re.findall(shipping, new_str)) or small_cost_30):
          return list(k) + ["shiping fee"]
        else: return list(k) + ["cost_product"]
    else: return [-1, -1, -1]
  

def additional_shipfee(sent):
    shipping = r'\b(ship|cod|phí|phi)\b'
    add_ship_pt = r'\b\d{2}\b'
    list_sq = findall_index(add_ship_pt, sent, 'shiping fee')
    ship_sq = []
    for e in list_sq:
        new_str = get_string_in_range(string = sent, neighbor=2, start=e[0], end=e[1])
        if re.search(shipping, new_str) is not None:
            ship_sq.append(e)
    return ship_sq

def label_full_string(input):
    input = input.lower()
    out = [-1, -1, -1]
    re = []
    string = input
    start = 0
    while True:
        check = label_shipping_and_cost(string)
        if check == out: break
        check[0] += start
        check[1] += start
        re.append(check)
        start = check[1]
        string = input[start:]
    freeship_sq = findall_index(r'((miễn|free)\s*)ship', input, 'shiping fee')
    add_shipfee = additional_shipfee(input)
    if len(freeship_sq) > 0:
        re.extend(freeship_sq)
    if len(add_shipfee) > 0:
        re.extend(add_shipfee)
    if re == []:
        return []
    return re

def remove_duplicate_entity(sent_entities, sent_len):
    a = [x for x in range(len(sent_entities))]
    a.sort(key = lambda x: sent_entities[x][1] - sent_entities[x][0], reverse=True)
    check = [False] * (sent_len + 1)
    final_entities = []
    for x in a:
        seq_label = sent_entities[x]
        if check[seq_label[0]] or check[seq_label[1]]: 
            continue
        for i in range(seq_label[0], seq_label[1]):
            check[i] = True
        final_entities.append(seq_label)
    return final_entities

def find_trie(a, s):
    cnt = 0
    cnt_space = 0
    root = 0
    res = 0
    for c in s:
        if (c not in list_char):
            break
        try:
            a[root][c]
        except:
            break
        cnt = cnt + 1
        root = a[root][c]
        if c == ' ':
            cnt_space += 1
        if a[root]['en']:
            res = cnt
        else:
            if c == ' ' and cnt_space >= 3:
                res = cnt - 1
    return res

def infer_ID_product(a, sent):
    res = []
    for i in range(len(sent)):
        sub = sent[i:]
        end_offset = find_trie(a, sub)
        if end_offset > 0:
            res.append([i, end_offset + i, 'ID_product'])
    res = remove_duplicate_entity(res, len(sent))
    return res

if __name__ == "__main__":
    print(MODELS_PATH)
