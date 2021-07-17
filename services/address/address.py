from services.address.common import api_address

def address_entity(sent, address_inp):
    result = api_address(sent, address_inp, 5)
    try:
        result = result['items'][0]
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(result)
    except:
        result = {'start_offset': 0, 'end_offset': 0, 'score': 0}

    # print(result)
    return [int(float(result['start_offset'])), int(float(result['end_offset'])), 'address',
            int(float(result['score']))]

### FIX PROBLEM WITH RE_SEARCH