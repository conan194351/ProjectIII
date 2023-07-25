import numpy as np
import json
def get_labels():
    with open('./data_src.json') as j:
        data=np.array(json.load(j),dtype=object)
        label=[item['name'] for item in data]
        return label