import json
import re
from unidecode import unidecode
import numpy as np
import time


def preprocess(sent):
    if re.search(r'^@@@@@ Khách \d:', sent) is not None:
        sent = re.sub(r'^@@@@@ Khách \d:', '              ', sent)
    elif re.search(r'^@@@@@ Khách \d\d:', sent) is not None:
        sent = re.sub(r'^@@@@@ Khách \d\d:', '               ', sent)
    elif re.search(r'^@@@@@ Khách \d\d\d:', sent) is not None:
        sent = re.sub(r'^@@@@@ Khách \d\d\d:', '                ', sent)
    elif re.search(r'^@@@@@ Khách \d\d\d\d:', sent) is not None:
        sent = re.sub(r'^@@@@@ Khách \d\d\d\d:', '                 ', sent)
    elif re.search(r'^@@@@@ Khách \d\d\d\d\d:', sent) is not None:
        sent = re.sub(r'^@@@@@ Khách \d\d\d\d\d:', '                  ', sent)

    sent = sent.lower()
    sent = unidecode(sent)
    # sent = re.sub(r'\s+', ' ', sent)
    sent = re.sub(r'[^\w\d/\-]', ' ', sent)
    # sent = re.sub(r'^\s|\s$', '', sent)
    return sent

class SortedList():
    def __init__(self, cost_function):
        self.lst = []
        self.weights = [0]*10
        self.cost_function = cost_function
    
    def append(self, new_value):
        min_weight = np.min(self.weights)
        min_pos = np.argmin(self.weights)
        if self.cost_function(new_value) > min_weight:
            self.lst = [i for i in self.lst if self.cost_function(i) > min_weight]
            self.weights[min_pos] = self.cost_function(new_value)
            self.lst.append(new_value)
        elif self.cost_function(new_value) == min_weight:
            self.lst.append(new_value)

    def get(self, n):
        result = sorted(self.lst, key = self.cost_function)
        try: return result[-n:]
        except: return result

def re_search(sent, enc):
    result = []
    sent = re.sub('-', ' ', sent)
    words = sent.split(' ')
    ### SEGMENT WORD-LEVEL
    for l in range(1, 4):
        for i in range(len(words) - l):
            word = ''.join(words[i:i + l + 1])
            size = len(word) + (l)
            try:
                start_ofs = len(' '.join(words[:i]))
                if len(' '.join(words[:i])) > 0: start_ofs += 1  ### plus 1 due to ' ' before the word
                end_ofs = start_ofs + size
                result += [(t, (start_ofs, end_ofs), size) for t in enc[word]]
            except KeyError:
                continue

    ### SEGMENT CHAR-LEVEL in WORD
    for i in range(len(words)):
        word = words[i]
        for l in range(1, len(word)):
            for j in [0, len(word) - l-1]:
                w = word[j:j + l + 1]
                size = len(w)
                try:
                    start_ofs = len(' '.join(words[:i])) + j
                    if len(' '.join(words[:i])) > 0: start_ofs += 1  ### plus 1 due to ' ' before the word
                    end_ofs = start_ofs + size
                    result += [(t, (start_ofs, end_ofs), size) for t in enc[w]]
                except:
                    continue

    return list(set(result))


### The functions ABC_lists() have similar steps:
### ### For each set of address, e.g. (w,d,p), (d,p)
### ### Check conditions, including:
### ### ### 1/ The lower is in the larger (e.g. a ward in a district)
### ### ### 2/ The lower appear before the larger (e.g. a ward is written before a district)
### Sort the list and get the top 100 score (score is computed based on the number of characters matched)
### Add empty set in case there are no large units (e.g. no province, no district but have wards)
### Return
def dp_lists(district_ls, province_ls, address_inp):
    district_in_province = address_inp['district_in_province']

    result = SortedList(lambda x:np.sum(np.sqrt(np.array(x)[:,2].astype(float))))
    zipped = [(i,j) for i in district_ls for j in province_ls]
    for (d,p) in zipped:
        cond_1 = d[0]=='' or district_in_province[d[0]]==p[0] or p[0]==''
        cond_2 = d[1]==None or p[1]==None or d[1][1] <= p[1][0]
        cond_3 = d[1]==None or p[1]==None or p[1][0] - d[1][1] <= 12
        if not cond_1 or not cond_2 or not cond_3: continue

        res_d, res_p = list(d), list(p)
        if res_d[0]!='' and res_p[0]=='': res_p[0] = district_in_province[res_d[0]]
        result.append([res_d, res_p])
    
    return result.get(10)


### return list of (ward, district, province)
### ### condition 1->3 check whether ward in district and district in province
### ### while condition 4->5 check whether they are in good order
def wdp_lists(ward_ls, dp_lists, address_inp):
    ward_in_district = address_inp['ward_in_district']
    ward_in_province = address_inp['ward_in_province']
    district_in_province = address_inp['district_in_province']

    result = SortedList(lambda x:np.sum(np.sqrt(np.array(x)[:,2].astype(float))))
    zipped = [(i,j,k) for i in ward_ls for (j,k) in dp_lists]
    for (w,d,p) in zipped:
        cond_1 = w[0]=='' or ward_in_district[w[0]]==d[0] or d[0]==''
        cond_2 = w[0]=='' or ward_in_province[w[0]]==p[0] or p[0]==''
        cond_3 = w[1]==None or d[1]==None or w[1][1] <= d[1][0]
        cond_4 = w[1]==None or p[1]==None or w[1][1] <= p[1][0]
        cond_5 = w[1]==None or d[1]==None or d[1][0] - w[1][1] <= 12
        cond_6 = w[1]==None or p[1]==None or (d[1]!=None or p[1][0] - w[1][1] <= 12)
        if not cond_1 or not cond_2 or not cond_3 \
            or not cond_4 or not cond_5 or not cond_6: continue
            
        res_w, res_d, res_p = list(w), list(d), list(p)
        if res_w[0]!='' and res_d[0]=='': res_d[0] = ward_in_district[res_w[0]]
        if res_d[0]!='' and res_p[0]=='': res_p[0] = district_in_province[res_d[0]]
        result.append([res_w, res_d, res_p])
        
    return result.get(10)


### return list of vill : (ward, district, province) provided
def vwdp_lists(vill_ls, wdp_ls, address_inp):
    vill_in_ward = address_inp['vill_in_ward']
    vill_in_district = address_inp['vill_in_district']
    vill_in_province = address_inp['vill_in_province']
    ward_in_district = address_inp['ward_in_district']
    district_in_province = address_inp['district_in_province']

    result = SortedList(lambda x:np.sum(np.sqrt(np.array(x)[:,2].astype(float))))
    sum = 0
    zipped = [(i,j) for i in vill_ls for j in wdp_ls]
    for (v, (w,d,p)) in zipped:
        cond_1 = v[0]=='' or vill_in_ward[v[0]]==w[0] or w[0]==''
        cond_2 = v[0]=='' or vill_in_district[v[0]]==d[0] or d[0]==''
        cond_3 = v[0]=='' or vill_in_province[v[0]]==p[0] or p[0]==''
        cond_4 = v[1]==None or w[1]==None or v[1][1] <= w[1][0]
        cond_5 = v[1]==None or d[1]==None or v[1][1] <= d[1][0]
        cond_6 = v[1]==None or p[1]==None or v[1][1] <= p[1][0]
        cond_7 = v[1]==None or w[1]==None or w[1][0] - v[1][1] <= 12
        cond_8 = v[1]==None or d[1]==None or (w[1]!=None or d[1][0] - v[1][1] <= 12)
        cond_9 = v[1]==None or p[1]==None or (w[1]!=None or d[1]!=None or p[1][0]-v[1][1] <= 12)
        if not cond_1 or not cond_2 or not cond_3 \
            or not cond_4 or not cond_5 or not cond_6\
            or not cond_7 or not cond_8 or not cond_9: continue

        res_v, res_w, res_d, res_p = list(v), list(w), list(d), list(p)
        if res_v[0]!='' and res_w[0]=='': res_w[0] = vill_in_ward[res_v[0]]
        if res_w[0]!='' and res_d[0]=='': res_d[0] = ward_in_district[res_w[0]]
        if res_d[0]!='' and res_p[0]=='': res_p[0] = district_in_province[res_d[0]]
        
        result.append([res_v, res_w, res_d, res_p])

    return result.get(10)


def svwdp_lists(street_ls, vwdp_ls, address_inp):
    street_in_district = address_inp['street_in_district']
    street_in_province = address_inp['street_in_province']

    result = SortedList(lambda x:np.sum(np.sqrt(np.array(x)[:,2].astype(float))))
    sum = 0
    # start = time()
    zipped = [(i,j) for i in street_ls for j in vwdp_ls]
    # print("Zip time:", time() - start)

    for (s, (v,w,d,p)) in zipped:
        # start = time()
        cond_1 = s[0]=='' or street_in_district[s[0]]==d[0] or d[0]==''
        cond_2 = s[0]=='' or street_in_province[s[0]]==p[0] or p[0]==''
        cond_3 = s[1]==None or v[1]==None or s[1][1] <= v[1][0]
        cond_4 = s[1]==None or w[1]==None or s[1][1] <= w[1][0]
        cond_5 = s[1]==None or d[1]==None or s[1][1] <= d[1][0]
        cond_6 = s[1]==None or p[1]==None or s[1][1] <= p[1][0]
        cond_7 = s[1]==None or v[1]==None or v[1][0] - s[1][1] <= 12
        cond_8 = s[1]==None or w[1]==None or (v[1]!=None or w[1][0]-s[1][1] <= 12)
        cond_9 = s[1]==None or d[1]==None or (v[1]!=None or w[1]!=None or d[1][0]-s[1][1] <= 12)
        cond_10 = s[1]==None or p[1]==None or (v[1]!=None or w[1]!=None or d[1]!=None or p[1][0]-s[1][1] <= 12)
        # cond_time += time()-start
        if not cond_1 or not cond_2 or not cond_3 \
            or not cond_4 or not cond_5 or not cond_6\
            or not cond_7 or not cond_8 or not cond_9 or not cond_10: continue
        # start = time()
        result.append([list(s),list(v), list(w), list(d), list(p)])

    return result.get(10)


def full_address_lists(sent, svwdp_ls):
    result_list = []
    for (s, v, w, d, p, score) in svwdp_ls:
        t = [i[1][0] for i in (s, v, w, d, p) if i[1] != None]
        print
        if len(t) == 0: continue
        words = sent[:t[0]].split(' ')[:-1]

        house_num = ('', None, 0)

        if len(words) >= 2:
            word = ' '.join([words[-2], words[-1]])
            cond_1 = re.search(r'\d+', words[-1]) is not None
            cond_2 = re.search(r'\d+', words[-2]) is not None
            cond_3 = re.search(r'^[\s\w\d/\-]+$', word) is not None

            if cond_1 and cond_2 and cond_3:
                start_ofs = len(' '.join(words[:-2]))
                if len(' '.join(words[:-2])) > 0: start_ofs += 1  ### plus 1 due to ' ' before the word
                end_ofs = start_ofs + len(word) + 1
                house_num = (word, (start_ofs, end_ofs), 0)

        if len(words) >= 1 and house_num[0] == '':
            cond_1 = re.search(r'\d+', words[-1]) is not None
            cond_2 = re.search(r'^[\w\d/\-]+$', words[-1]) is not None
            if cond_1 and cond_2:
                start_ofs = len(' '.join(words[:-1]))
                if len(' '.join(words[:-1])) > 0: start_ofs += 1  ### plus 1 due to ' ' before the word
                end_ofs = start_ofs + len(words[-1])
                house_num = (words[-1], (start_ofs, end_ofs), 0)

        result_list.append((house_num, s, v, w, d, p, score))

    return result_list


def get_address(sent, address_inp, output_amount=5):
    street_enc = address_inp['street_enc']
    vill_enc = address_inp['vill_enc']
    ward_enc = address_inp['ward_enc']
    district_enc = address_inp['district_enc']
    province_enc = address_inp['province_enc']

    # sent = preprocess(sent)

    street_ls = re_search(sent, street_enc)
    vill_ls = re_search(sent, vill_enc)
    ward_ls = re_search(sent, ward_enc)
    district_ls = re_search(sent, district_enc)
    province_ls = re_search(sent, province_enc)

    street_ls += [('', None, 0)]
    vill_ls += [('', None, 0)]
    ward_ls += [('', None, 0)]
    district_ls += [('', None, 0)]
    province_ls += [('', None, 0)]

    # start = time()
    dp_ls = dp_lists(district_ls, province_ls, address_inp)
    # print("DP TIME:", time()-start)
    # start = time()
    wdp_ls = wdp_lists(ward_ls, dp_ls, address_inp)
    # print("WDP TIME:", time()-start)
    # start = time()
    vwdp_ls = vwdp_lists(vill_ls, wdp_ls, address_inp)
    # print("VWDP TIME:", time()-start)
    # start = time()
    svwdp_ls = svwdp_lists(street_ls, vwdp_ls, address_inp)
    # print("SVWDP TIME:", time()-start)

    # start = time()
    tmp_ls = []
    for (s, v, w, d, p) in svwdp_ls:
        score = np.sum([np.sqrt(i[2]) for i in (s, v, w, d, p)])
        tmp_ls.append([s, v, w, d, p, score])
    tmp_sorted = sorted(tmp_ls, key=lambda x: x[5])
    # print("SORTING TIME:", time() - start)

    # start= time()
    try:
        full_a_ls = full_address_lists(sent, tmp_sorted[-output_amount:])
    except:
        full_a_ls = full_address_lists(sent, tmp_sorted)
    # print("FULL LS TIME:", time()-start)

    # print(len(street_ls), len(vill_ls), len(ward_ls), len(district_ls), len(province_ls))
    # print(len(wdp_ls), len(vwdp_ls), len(svwdp_ls), len(full_a_ls))
    return full_a_ls


### result_list is a list of
### (house_num,street,vill,ward,district,province,score)
### besides, they store Code or ID instead of Name
def expected_output(result_list, sent, address_inp):
    street_dec = address_inp['street_dec']
    vill_dec = address_inp['vill_dec']
    ward_dec = address_inp['ward_dec']
    district_dec = address_inp['district_dec']
    province_dec = address_inp['province_dec']

    # sent = preprocess(sentence)
    result = {'p': sent, 'p_normalize': sent, 'items': []}
    for address in result_list:
        result['items'].append({'score': None, 'address': None, 'phoneNumber': None, 'shortAddress': None, \
                                'street': None, 'village': None, 'wardName': None, 'wardCode': None, \
                                'districtName': None, 'districtCode': None, 'cityName': None, 'cityCode': None, \
                                'start_offset':None, 'end_offset': None})

        result['items'][-1]['score'] = str(address[-1])

        ### province
        if address[5][2] != 0:
            result['items'][-1]['cityCode'] = str(address[5][0])
            result['items'][-1]['cityName'] = province_dec[address[5][0]]

        ### district
        if address[4][2] != 0:
            result['items'][-1]['districtCode'] = str(address[4][0])
            result['items'][-1]['districtName'] = district_dec[address[4][0]]

        ### ward
        if address[3][2] != 0:
            result['items'][-1]['wardCode'] = str(address[3][0])
            result['items'][-1]['wardName'] = ward_dec[address[3][0]]

        ### village
        if address[2][2] != 0:
            result['items'][-1]['village'] = str(vill_dec[address[2][0]])

        ### street
        if address[1][2] != 0:
            result['items'][-1]['street'] = str(street_dec[address[1][0]])

        ### house number
        house_num = address[0][0]

        ### Short Address
        tmp = [house_num] + [result['items'][-1][z] for z in ['street', 'village']]
        tmp = [z for z in tmp if z is not None and len(z) > 0]
        if len(' '.join(tmp)) > 1:
            result['items'][-1]['shortAddress'] = ' '.join(tmp)

        ### full address
        tmp = [result['items'][-1][z] for z in ['shortAddress', 'wardName', 'districtName', 'cityName']]
        tmp = [z for z in tmp if z is not None and len(z) > 0]
        result['items'][-1]['address'] = ' '.join(tmp)

        # print(">>>>>>>>>>>>>>>>>>>> BACH DEBUG Address")
        # print(address)
        offset_ls = [i[1] for i in address[:-1] if i[1] is not None]
        result['items'][-1]['start_offset'] =  offset_ls[0][0]
        result['items'][-1]['end_offset'] = offset_ls[-1][1]

    # print(result)
    result['items'].reverse()
    return result


def api_address(sent, address_inp, output_amount=5):
    sent = preprocess(sent)
    result_list = get_address(sent, address_inp, output_amount)
    result = expected_output(result_list, sent, address_inp)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(result['p'])
    # print(result['items'][0])
    return result