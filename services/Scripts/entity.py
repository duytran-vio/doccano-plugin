from vncorenlp import VnCoreNLP
from os import path
import re
import pandas as pd

BASEDIR = path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
# BASEDIR = 'D:\\GitHub\\VnCoreNLP'
vncorenlp_file = path.join(BASEDIR,path.join('VnCoreNLP','VnCoreNLP-1.1.1.jar'))

MODELS_PATH = path.join(
    path.dirname(path.dirname(path.abspath(__file__))),
    'models'
)

### COLOR_PRODUCT
df_colors = pd.read_csv(path.join(MODELS_PATH, 'colors.csv'), header=None)
colors = df_colors[0].tolist()
pt_color = r'((?<=^)|(?<=\s))(' + '|'.join(colors) + r')((\s(đậm|dam|nhạt|nhat)*)|(?=\s|[^a-z]|$))'
###------------------------------------------

### COST_PRODUCT
dong_pt = r'đồng|dong|đ|dog|VND|VNĐ'
cost_pt = r'\d+\s*(k|tr((iệ|ie)u)*(\s{0:}|\s*\d*)*|ng[a|à]n(\s{0:}|\s*\d*)*|t[ỉiỷy](\s{0:}|\s*\d*)*|{0:})'.format(dong_pt)
# cost_ques = r'b(ao\s)*n(hi[e|ê]*u)*'
cost_pt_sum = '{}'.format(cost_pt)
###------------------------------------------

### AMOUNT_PRODUCT
df_amount_suf = pd.read_csv(path.join(MODELS_PATH, 'amount_suf.csv'), header = None)
amount_suf = df_amount_suf[0].tolist()
product_pt = '[á|a]o|qu[a|ầ]n|v[a|á]y|đầm|dam|t[ú|u]i|n[ó|o]n|m[u|ũ]'
amount_pt = r'(\d+-)*\d+\s*(' + '|'.join(amount_suf) + r')((\s({})*)|(?=[^a-z]|$))'.format(product_pt) 
amount_pt_2 = r'(\d+-)*\d+\s*({})'.format(product_pt)
amount_pt_sum = r'{0:}|{1:}'.format(amount_pt, amount_pt_2)
###------------------------------------------

### MATERIAL_PRODUCT
df_material = pd.read_csv(path.join(MODELS_PATH, 'material.csv'), header = None)
material = df_material[0].tolist()
pt_material = r'((ch[a|ấ]t(\sli[e|ệ]u)*|lo[a|ạ]i)\s)*(' + '|'.join(material) + r')(\sc[u|ứ]ng|\sm[e|ề]m)*'
###------------------------------------------

### list of pattern
list_entity_using_regex = ['phone', 'weight customer', 'height customer',
                             'color_product'
                            ]
pattern_list = {
    'phone': [
        r'[0-9]{4}\.*[0-9]{3}\.*[0-9]{2,}'
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
        r'(size|sai|sz)\s(\d*[smlx]*[SMLX]*)'
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

def label_entity(sentences):
    sents_entity = [[] for i in range(len(sentences))]
    # ner_entity = [[] for i in range(len(sentences))]

    ## Use ner to get Id member entity
    # with VnCoreNLP(address='http://127.0.0.1', port=9000) as vncorenlp:
    #     for i in range(len(sentences)):
    #         sent = sentences[i]
    #         # print(sent)
    #         list_Id_member_sq = infer_Id_member(sent,vncorenlp)
    #         list_Id_member_sq = reduce_label(list_Id_member_sq, sent.find(':'))
    #         # print(list_Id_member_sq)
    #         ner_entity[i] = list_Id_member_sq

    ## Use regex
    for entity in list_entity_using_regex:
        for i in range(len(sentences)):
            sent = sentences[i].lower()
            list_entity_sq = get_entity_sq_from_list_pt(pattern_list[entity], sent, entity)
            list_entity_sq = join_continuous_sq(list_entity_sq, sent)
            list_entity_sq = reduce_label(list_entity_sq, sent.find(':'))
            if len(list_entity_sq) > 0:
                sents_entity[i].extend(list_entity_sq)

    ## Merge Id member to sents_entity
    # for i in range(len(sentences)):
    #     sents_entity[i] = merge(ner_entity[i], sents_entity[i])

    return sents_entity

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

if __name__ == "__main__":
    print(MODELS_PATH)
