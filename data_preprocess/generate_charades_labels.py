import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math
import csv

caption_dir = "/home/zenglh/charades"
caption_train_file = 'charades_sta_train.txt'
caption_val_file = 'charades_sta_test.txt'
csv_train_file = 'Charades_v1_train.csv'
csv_val_file = 'Charades_v1_test.csv'
output_file ='charades_caption.json'

def generate_feature_caption(caption, csv_file, dataset):
    csv_reader = csv.reader(csv_file, delimiter=',')
    durations = {}
    for row in csv_reader:
        durations[row[0]] = row[-1]
    caption_json = {}
    for line in caption:
        lines = line.split('##')
        sentenses = lines[1][:-1]
        others = lines[0].split(' ')
        video_name = others[0]
        timestamp = [float(others[1]), float(others[2])]
        if video_name not in caption_json:
            caption_json[video_name] = {'sentences':[sentenses], 'duration':durations[video_name], 
                'timestamps':[timestamp], 'subset':dataset}
        else:
            caption_json[video_name]['sentences'].append(sentenses)
            caption_json[video_name]['timestamps'].append(timestamp)
    return caption_json
    

def main():
    with open(os.path.join(caption_dir, caption_train_file),'r',encoding='utf-8') as caption:
        with open(os.path.join(caption_dir, csv_train_file)) as csv_file:
            caption_json_train = generate_feature_caption(caption, csv_file, 'train')
    with open(os.path.join(caption_dir, caption_val_file),'r',encoding='utf-8') as caption:
        with open(os.path.join(caption_dir, csv_val_file)) as csv_file:
            caption_json_val = generate_feature_caption(caption, csv_file, 'validation')
    caption_json_train.update(caption_json_val)
    with open(os.path.join(caption_dir, output_file),'w',encoding='utf-8') as caption:
        json.dump(caption_json_train, caption)

if __name__ == "__main__":
    main()