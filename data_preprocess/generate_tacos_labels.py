import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math
import csv
import cv2

caption_dir = "/home/zenglh/TACoS"
video_dir = "/home/zenglh/videos"
csv_file_name = 'index.tsv'
output_file ='tacos_caption.json'

def generate_feature_caption(csv_file):
    csv_reader = csv.reader(csv_file, delimiter='	')
    durations = {}
    frames_rate = {}
    caption_json = {}
    for row in csv_reader:
        if len(durations.keys()) % 4 == 0:
            dataset = 'validation'
        else:
            dataset = 'train'
        video_name = row[2] + '.avi'
        if video_name not in frames_rate or video_name not in durations:
            cap = cv2.VideoCapture(os.path.join(video_dir,video_name))
            frames_rate[video_name] = cap.get(5)
            durations[video_name] = float(cap.get(7)) / float(cap.get(5))
            cap.release()
        timestamp = [float(row[3]) / frames_rate[video_name], float(row[4]) / frames_rate[video_name]]
        sentenses = row[1]
        if video_name not in caption_json:
            caption_json[video_name] = {'sentences':[sentenses], 'duration':durations[video_name], 
                'timestamps':[timestamp], 'subset':dataset}
        else:
            caption_json[video_name]['sentences'].append(sentenses)
            caption_json[video_name]['timestamps'].append(timestamp)

    return caption_json
    

def main():
    with open(os.path.join(caption_dir, csv_file_name),'r',encoding='utf-8') as csv_file:
        caption_json = generate_feature_caption(csv_file)
    with open(os.path.join(caption_dir, output_file),'w',encoding='utf-8') as caption:
        json.dump(caption_json, caption)

if __name__ == "__main__":
    main()