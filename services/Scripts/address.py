import json 
import re
from unidecode import unidecode
import numpy as np
import time
import pickle

def address_entity(sent, address_inp):
    enc, dec, id2w, disown, ward_dis = address_inp
    result_list,sent = express_addr(sent, enc, dec, disown, ward_dis)
    
    try:
        result = expected_output(result_list, id2w, sent)['items'][0]
    except:
        result = {'start_offset': 0, 'end_offset': 0, 'score': 0}
    
    # print(result)
    return [int(float(result['start_offset'])), int(float(result['end_offset'])), 'address', int(float(result['score']))]
    # return [0, 0, 'address']

def preprocess(sentence):
    sentence = sentence.lower()
    # sentence = re.sub('\xa0|[^\w\d\\/]', ' ', sentence)
    # sentence = re.sub(r'\s+', ' ', sentence)
    # sentence = re.sub(r'^\s+|\s+$', '', sentence) 
    return sentence

def re_search(sent, dec, info='DistrictCode'):
    result, tmp = [], []
    sent = preprocess(sent)
    words = sent.split(' ')
    lens = [len(i) for i in words]
    lens = [np.sum(lens[0:i])+i for i in range(len(words))]
    for l in range(4):
        for i in range(len(words)-l):
        # for i in range(len(sent)-l):
            word = ' '.join([w for w in words[i:i+l+1]])
            word = re.sub('\s+', ' ', word)
            # word = sent[i:i+l+1]
            try:
                if word not in tmp:
                    result += [(dis, (lens[i], lens[i]+len(word), len(word) ), len(word) ) for dis in dec[word][info] ]
                    # result += [(dis, (i, i+l ), len(word) ) for dis in dec[word][info] ]
                    # tmp.append(word)
            except KeyError:
                continue
    return list(set(result))

def mini_idx(l):
    result = 9999
    for i in range(len(l)):
        try:
            result = min(result, l[i][1][0])
        except:
            pass
    return result

def max_idx(l):
    result = -1
    for i in range(len(l)):
        try:
            result = max(result, l[i][1][1])
        except:
            pass
    return result

def handle_non_district(sent, dec, ward_dis):
    street_dec, vill_dec, ward_dec, district_dec, province_dec = dec
    d_list = re_search(sent, ward_dec, info='WardCode')
    result = []
    for i in d_list:
        result += [[[], [], [i], [(j, (i[1][1], i[1][1]))], []] for j in ward_dis[i[0]]]
    return result


def get_address(sent, enc, dec, disown, ward_dis):
    street_enc, vill_enc, ward_enc, district_enc, province_enc = enc
    street_dec, vill_dec, ward_dec, district_dec, province_dec = dec
    dis_street, dis_vill, dis_ward, dis_province = disown

    d_list = re_search(sent, district_dec)
    d_list = [[[], [], [], [i], []] for i in d_list]
    d_list += handle_non_district(sent, dec, ward_dis)

    mini = [i[3][0][1][0] if len(i[2])==0 else i[2][0][1][0] for i in d_list]

    for i in range(len(d_list)):
        district = d_list[i][3][0]
        d = district[0]
        
        all_matches = set(re_search(sent, province_dec, 'ProvinceCode'))
        for ret in all_matches:
            if ret[0] in dis_province[d] and ret[1][0] > mini[i]:
                d_list[i][4] += [ret]
        if len(d_list[i][4]) < 1:
            for province in dis_province[d]:
                d_list[i][4] += [(province, (district[1][1], district[1][1]) )]

        tmp_min = -1 
        all_matches = set(re_search(sent, ward_dec, 'WardCode'))
        for ret in all_matches:
            if ret[0] in dis_ward[d] and ret[1][1] <= mini[i]:
                d_list[i][2] += [ret]
                tmp_min = max(mini_idx(d_list[i][2]), tmp_min)
        if tmp_min >= 0:
            mini[i] = tmp_min

        tmp_min = -1
        all_matches = set(re_search(sent, vill_dec, 'LangID'))
        for ret in all_matches:
            if ret[0] in dis_vill[d] and ret[1][1] <= mini[i]:
                d_list[i][1] += [ret]
                tmp_min = max(mini_idx(d_list[i][1]), tmp_min)
        if tmp_min >= 0:
            mini[i] = tmp_min

        all_matches = set(re_search(sent, street_dec, 'StreetID'))
        for ret in all_matches:
            if ret[0] in dis_street[d] and ret[1][1] <= mini[i]:
                d_list[i][0] += [ret]

    return d_list

def remove_nearly_empty(lists, thresh=2):
    result = [l for l in lists if len([sub_l for sub_l in l if len(sub_l)>0]) > thresh]
    return result

def extract_results(addr_list):
    results = []
    for addr in addr_list:
        for i in range(len(addr)):
            if len(addr[i]) == 0:
                addr[i].append([''])
        for i0 in addr[0]:
            for i1 in addr[1]:
                for i2 in addr[2]:
                    for i3 in addr[3]:
                        for i4 in addr[4]:
                            for i5 in addr[5]:
                                results.append([i0, i1, i2, i3, i4, i5])
    return results

def express_addr(sent, enc, dec, disown, ward_dis):
    sent = unidecode(sent.lower())
    # print(sent)
    # print(sent)
    results = []
    addr_list = get_address(sent, enc, dec, disown, ward_dis)
    for addr in addr_list:
        for i in range(3):
            i = addr[i]
            if len(i) > 0:
                first = mini_idx(i)
                if first > 0:
                    addr_tkn = sent[0:first].split(' ')
                break
        one, second = '', ''
        try:
            one = addr_tkn[-1]
            second = addr_tkn[-2]
        except:
            pass
        if re.search(r'^[/\\a-zA-Z]*\d+[/\\\da-zA-Z]*$', one) is not None:
            addr = [[[one]]] + addr
        elif re.search(r'^[/\\a-zA-Z]*\d+[/\\\da-zA-Z]*$', second) is not None:
            addr = [[(second, (first-len(one)-len(second)-1, -1) )]] + addr
        else:
            addr = [[['']]] + addr
        results.append(addr)

    results = remove_nearly_empty(results)
    # print(results)
    return (extract_results(results), sent)

def expected_output(result_list, id2w, sent):
    street_id2w, vill_id2w, ward_id2w, district_id2w, province_id2w = id2w
    result = {'p': None, 'p_normalize': sent, 'items': []}
    for i in result_list:
        result['items'].append({'score': 0, 'address': None, 'phoneNumber': None, 'shortAddress': None, \
                                'street': None, 'village': None, 'wardName': None, 'wardCode': None, \
                                'districtName': None, 'districtCode': None, 'cityName': None, 'cityCode': None,  \
                                'start_offset':None, 'end_offset': None})
        try:
            result['items'][-1]['start_offset'] = str(mini_idx(i))
        except:
            pass

        try:
            result['items'][-1]['end_offset'] = str(max_idx(i))
        except:
            pass

        try:
            result['items'][-1]['score'] = get_score(i)
        except:
            pass

        try:
            result['items'][-1]['cityCode'] = int(i[5][0])
            result['items'][-1]['cityName'] = province_id2w[ i[5][0] ]
        except:
            pass 

        try:
            result['items'][-1]['districtCode'] = int(i[4][0])
            result['items'][-1]['districtName'] = district_id2w[ i[4][0] ]
        except:
            pass

        try:
            result['items'][-1]['wardCode'] = int(i[3][0])
            result['items'][-1]['wardName'] = ward_id2w[i[3][0]]
        except:
            pass
    
        try:
            result['items'][-1]['village'] = vill_id2w[i[2][0]]
        except:
            pass

        try:
            result['items'][-1]['street'] = street_id2w[i[1][0]]
        except:
            pass
        
        home_addr = None
        try:
            home_addr = i[0][0]
        except:
            pass

        try:
            tmp = [home_addr] + [result['items'][-1][z] for z in ['street', 'village']]
            tmp = [z for z in tmp if z is not None and len(z)>0]
            if len(' '.join(tmp) ) > 1:
                result['items'][-1]['shortAddress'] = ' '.join(tmp)
        except:
            pass

        try:
            tmp = [result['items'][-1][z] for z in ['shortAddress', 'wardName', 'districtName', 'cityName']]
            tmp = [z for z in tmp if z is not None and len(z)>0]
            result['items'][-1]['address'] = ' '.join(tmp)
        except:
            pass
    
    # print(result)
    result['items'].sort(key=sortingFunc, reverse=True)
    return result

def get_score(e):
    result = 0
    for i in e:
        if len(i) > 2:
            result += i[2]
    return result

def sortingFunc(e):
    return e['score']

def prepare_enc():
    street_enc = pickle.load(open('services/Scripts/savedfiles/encoder/street_enc.pkl', 'rb'))
    vill_enc = pickle.load(open('services/Scripts/savedfiles/encoder/vill_enc.pkl', 'rb'))
    ward_enc = pickle.load(open('services/Scripts/savedfiles/encoder/ward_enc.pkl', 'rb'))
    district_enc = pickle.load(open('services/Scripts/savedfiles/encoder/district_enc.pkl', 'rb'))
    province_enc = pickle.load(open('services/Scripts/savedfiles/encoder/province_enc.pkl', 'rb'))
    print("Encoders loaded.")
    return street_enc, vill_enc, ward_enc, district_enc, province_enc

def prepare_dec():
    street_dec = pickle.load(open('services/Scripts/savedfiles/decoder/street_dec.pkl', 'rb'))
    vill_dec = pickle.load(open('services/Scripts/savedfiles/decoder/vill_dec.pkl', 'rb'))
    ward_dec = pickle.load(open('services/Scripts/savedfiles/decoder/ward_dec.pkl', 'rb'))
    district_dec = pickle.load(open('services/Scripts/savedfiles/decoder/district_dec.pkl', 'rb'))
    province_dec = pickle.load(open('services/Scripts/savedfiles/decoder/province_dec.pkl', 'rb'))
    print("Decoders loaded.")
    return street_dec, vill_dec, ward_dec, district_dec, province_dec

def prepare_id2w():
    street_id2w = pickle.load(open('services/Scripts/savedfiles/id2w/street_id2w.pkl', 'rb'))
    vill_id2w = pickle.load(open('services/Scripts/savedfiles/id2w/vill_id2w.pkl', 'rb'))
    ward_id2w = pickle.load(open('services/Scripts/savedfiles/id2w/ward_id2w.pkl', 'rb'))
    district_id2w = pickle.load(open('services/Scripts/savedfiles/id2w/district_id2w.pkl', 'rb'))
    province_id2w = pickle.load(open('services/Scripts/savedfiles/id2w/province_id2w.pkl', 'rb'))
    print("ID2Ws loaded.")
    return street_id2w, vill_id2w, ward_id2w, district_id2w, province_id2w

def prepare_disown():
    dis_street = pickle.load(open('services/Scripts/savedfiles/disown/dis_street.pkl', 'rb'))
    dis_vill = pickle.load(open('services/Scripts/savedfiles/disown/dis_vill.pkl', 'rb'))
    dis_ward = pickle.load(open('services/Scripts/savedfiles/disown/dis_ward.pkl', 'rb'))
    dis_province = pickle.load(open('services/Scripts/savedfiles/disown/dis_province.pkl', 'rb'))
    print("Districts' dependants loaded.")
    return dis_street, dis_vill, dis_ward, dis_province

def prepare_ward_s_district(dis_ward):
    result = {}
    for i in dis_ward.keys():
        for j in dis_ward[i]:
            try:
                result[j].append(i)
            except:
                result[j] = [i]
    return result