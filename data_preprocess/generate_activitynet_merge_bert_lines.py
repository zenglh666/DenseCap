import json
import os
import h5py
import numpy as np
import sys
from io import StringIO

label_file = '/home/zenglh/activitynet/activitynet_caption2.json'
feature_file = '/home/zenglh/activitynet/activitynet_sentences_line2.jsonl'
output_file = '/home/zenglh/activitynet/activitynet_bert2.hdf5'

def list_features(feature):
    syntaxes = feature['features']
    fl = []
    for s in syntaxes:
        f = s['layers'][-1]['values']
        fl.append(np.expand_dims(f, 0))
    return np.concatenate(fl, 0)


with open(os.path.join(label_file), "r") as fl:
    captions = json.load(fl)

feature_list = []
with open(os.path.join(feature_file), "r") as fb:
    feature_lines = fb.read().splitlines()
    for line in feature_lines:
        feature = json.load(StringIO(line))
        feature = list_features(feature)
        feature_list.append(feature)

feature_dict = {}
count = 0
keys = sorted(captions.keys())
for k in keys:
    feature_dict[k] = []
    for i in range(len(captions[k]["sentences"])):
        feature_dict[k].append(feature_list[count])
        count += 1

print(count)

with h5py.File(os.path.join(output_file),'w') as features_hdf5:
    for k, v in feature_dict.items():
        grp = features_hdf5.create_group(k)
        for i in range(len(v)):
            grp.create_dataset(str(i), data=v[i].astype(np.float32))