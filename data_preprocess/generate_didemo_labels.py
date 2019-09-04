import numpy as np
import h5py
import json
import sys
import os 
import math

caption_dir = '/home/zenglh/didemo'
caption_train_file = 'train_data.json'
caption_val_file = 'val_data.json'
caption_test_file = 'test_data.json'
output_file = 'didemo_caption.json'


def generate_proposal(caption_json, dataset):
    caption_json = json.load(caption_json)

    propsoal_json = {}
    for value in caption_json:
        proposal = {}
        proposal["duration"] = 30.
        proposal["subset"] = dataset
        proposal["timestamps"] = []
        for ti in value["times"][:4]:
            proposal["timestamps"].append([float(ti[0]) * 5., float(ti[1] + 1) * 5.])
        proposal["sentences"] = [value["description"]] * len(proposal["timestamps"])
        key = value["video"]

        if key not in propsoal_json:
            propsoal_json[key] = proposal
        else:
            propsoal_json[key]["timestamps"].extend(proposal["timestamps"])
            propsoal_json[key]["sentences"].extend(proposal["sentences"])
    return propsoal_json

def main():
    with open(os.path.join(caption_dir, output_file), "w") as f:
        with open(os.path.join(caption_dir, caption_train_file),'r',encoding='utf-8') as caption:
            merge_train = generate_proposal(caption, 'train')
        with open(os.path.join(caption_dir, caption_val_file),'r',encoding='utf-8') as caption:
            merge_val = generate_proposal(caption, 'train')
        with open(os.path.join(caption_dir, caption_test_file),'r',encoding='utf-8') as caption:
            merge_test = generate_proposal(caption, 'validation')
        for k in merge_val.keys():
            if k in merge_train.keys():
                raise ValueError("two dataset")
        merge_train.update(merge_val)
        for k in merge_test.keys():
            if k in merge_train.keys():
                raise ValueError("two dataset")
        merge_train.update(merge_test)
        json.dump(merge_train, f)
            

if __name__ == "__main__":
    main()