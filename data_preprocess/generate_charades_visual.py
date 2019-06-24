import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math
from sklearn.decomposition import IncrementalPCA

input_dir = '/home/zenglh/charades/Charades_v1_features_rgb'
input_dir_flow = '/home/zenglh/charades/Charades_v1_features_flow'
output_file = '/home/zenglh/charades/charades_visual.hdf5'
output_file2 = '/home/zenglh/charades/charades_visual_pca.hdf5'

def main():
    transformer = IncrementalPCA(n_components=500, batch_size=20000)
    partial_list = []
    current_size = 0
    finish_size = 0
    with h5py.File(output_file,'w') as output_h5:
        for i in os.listdir(input_dir):
            temp_dir = os.path.join(input_dir, i)
            temp_dir_flow = os.path.join(input_dir_flow, i)
            if os.path.isdir(temp_dir):
                feature_rgb = []
                feature_flow = []
                for f in sorted(os.listdir(temp_dir)):
                    temp_file = os.path.join(temp_dir, f)
                    temp_file_flow = os.path.join(temp_dir_flow, f)
                    if os.path.isfile(temp_file) and os.path.isfile(temp_file_flow):
                        with open(temp_file, 'r') as file:
                            numbers = file.read().split(' ')
                        feature_rgb.append(numbers)
                        with open(temp_file_flow, 'r') as file:
                            numbers = file.read().split(' ')
                        feature_flow.append(numbers)
                feature_rgb = np.array(feature_rgb).astype(np.float32)
                feature_flow = np.array(feature_flow).astype(np.float32)
                feature = np.concatenate([feature_rgb, feature_flow], axis=1)
                output_h5.create_dataset(i, data=feature)
                partial_list.append(feature)
                current_size += feature.shape[0]

                if current_size > transformer.batch_size:
                    partial = np.concatenate(partial_list, axis=0)
                    transformer.partial_fit(partial)
                    finish_size += current_size
                    print('finish samples %d', finish_size)
                    partial_list = []
                    current_size = 0
                

    with h5py.File(output_file2,'w') as output_h5:
        with h5py.File(output_file,'r') as input_h5:
            for k in input_h5.keys():
                output_h5.create_dataset(k, data=transformer.transform(input_h5[k]).astype(np.float32))
       

if __name__ == "__main__":
    main()