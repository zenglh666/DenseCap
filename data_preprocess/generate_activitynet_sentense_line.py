import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math

input_file = '/home/zenglh/activitynet/activitynet_caption.json'
output_file = '/home/zenglh/activitynet/activitynet_sentences_line.txt'


def main():
    with open(input_file,'r',encoding='utf-8') as f:
        caption_json = json.load(f)

    with open(output_file,'w',encoding='utf-8') as f:
        keys = caption_json.keys()
        keys = sorted(keys)
        lines = []
        for k in keys:
            for sentence in caption_json[k]["sentences"]:
                lines.append(sentence.replace('\n', '').replace('\r', '').strip() + '\n')
        f.writelines(lines)
            

if __name__ == "__main__":
    main()