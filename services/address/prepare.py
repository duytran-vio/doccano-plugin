from unidecode import unidecode
import numpy as np
import json
import re


### get list of dicts from json
def dicts_from_json(json_path='update_districts.json'):
    with open(json_path, 'r') as f:
        return_dicts = json.load(f)
    return return_dicts


### get list of dicts from json
def dicts_from_json(json_path='update_districts.json'):
    with open(json_path, 'r') as f:
        return_dicts = json.load(f)
    return return_dicts

### tags (preprocessed) to codes (list of codes)
def build_enc(dicts, code):
    return_dict = {}
    for d in dicts:
        for tag in d['tags']:
            try: return_dict[tag].append( d[code] ) 
            except: return_dict[tag] = [ d[code] ]
    return return_dict

### code to name (full name)
def build_dec(dicts, code, name):
    return_dict = {}
    for d in dicts:
        return_dict[ d[code] ] = d[name]
    return return_dict

### return dictionary of Acode in Bcode
### for example a District is in a Province
### because take the "code" so dict is 1 key - 1 value
def AinB(dicts, Acode, Bcode):
    return_dict = {}
    for d in dicts:
        return_dict[ d[Acode] ] = d[Bcode]
    return return_dict

def preparation(folder='json'):
    print("Start preparing inputs for address....")
    address_inp = {}

    ### get dictionaries from json files in folder
    
    province_dicts = dicts_from_json(folder + '/update_provinces.json')
    district_dicts = dicts_from_json(folder + '/update_districts.json')
    ward_dicts = dicts_from_json(folder + '/update_wards.json')
    vill_dicts = dicts_from_json(folder + '/update_villages.json')
    street_dicts = dicts_from_json(folder + '/update_streets.json')

    address_inp['province_enc'] = build_enc(province_dicts, 'ProvinceCode')
    address_inp['province_dec'] = build_dec(province_dicts, 'ProvinceCode', 'Province')

    address_inp['district_enc'] = build_enc(district_dicts, 'DistrictCode')
    address_inp['district_dec'] = build_dec(district_dicts, 'DistrictCode', 'District')

    address_inp['ward_enc'] = build_enc(ward_dicts, 'WardCode')
    address_inp['ward_dec'] = build_dec(ward_dicts, 'WardCode', 'Ward')

    address_inp['vill_enc'] = build_enc(vill_dicts, 'LangID')
    address_inp['vill_dec'] = build_dec(vill_dicts, 'LangID', 'Lang')

    address_inp['street_enc'] = build_enc(street_dicts, 'StreetID')
    address_inp['street_dec'] = build_dec(street_dicts, 'StreetID', 'StreetName')

    address_inp['street_in_district'] = AinB(street_dicts, 'StreetID', 'DistrictCode')
    address_inp['street_in_province'] = AinB(street_dicts, 'StreetID', 'ProvinceCode')
    address_inp['ward_in_district'] = AinB(ward_dicts, 'WardCode', 'DistrictCode')
    address_inp['ward_in_province'] = AinB(ward_dicts, 'WardCode', 'ProvinceCode')
    address_inp['district_in_province'] = AinB(district_dicts, 'DistrictCode', 'ProvinceCode')
    address_inp['vill_in_ward'] = AinB(vill_dicts, 'LangID', 'WardCode')
    address_inp['vill_in_district'] = AinB(vill_dicts, 'LangID', 'DistrictCode')
    address_inp['vill_in_province'] = AinB(vill_dicts, 'LangID', 'ProvinceCode')

    print("Complete preparing inputs for address.")

    return address_inp