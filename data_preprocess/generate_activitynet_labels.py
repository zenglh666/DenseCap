import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math

caption_dir = '/home/zenglh/activitynet/captions'
caption_train_file = 'train.json'
caption_train_id_file = 'train_ids.json'
caption_val_file = 'val_1.json'
caption_val_id_file = 'val_ids.json'
output_file = 'activitynet_caption.json'


def generate_proposal(caption_json, dataset):
    caption_json = json.load(caption_json)
    keys = caption_json.keys()

    propsoal_json = {}
    for key, value in caption_json.items():
        proposal = {}
        proposal["duration"] = value["duration"]
        proposal["subset"] = dataset
        proposal["timestamps"] = value["timestamps"]
        proposal["sentences"] = value["sentences"]
        propsoal_json[key] = proposal
    return propsoal_json

def main():
    with open(os.path.join(caption_dir, output_file), "w") as f:
        with open(os.path.join(caption_dir, caption_train_file),'r',encoding='utf-8') as caption:
            merge_train = generate_proposal(caption, 'train')
        with open(os.path.join(caption_dir, caption_val_file),'r',encoding='utf-8') as caption:
            merge_val = generate_proposal(caption, 'validation')
        merge_train.update(merge_val)
        json.dump(merge_train, f)
            

if __name__ == "__main__":
    main()