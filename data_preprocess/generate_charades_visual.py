import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math

input_dir = '/home/zenglh/charades/Charades_v1_features_rgb'
output_file = '/home/zenglh/charades/charades_visual.hdf5'


def main():
    with h5py.File(output_file,'w') as output_h5:
        for i in os.listdir(input_dir):
            temp_dir = os.path.join(input_dir, i)
            if os.path.isdir(temp_dir):
                feature = []
                for f in sorted(os.listdir(temp_dir)):
                    temp_file = os.path.join(temp_dir, f)
                    if os.path.isfile(temp_file):
                        with open(temp_file, 'r') as file:
                            numbers = file.read().split(' ')
                        feature.append(np.array([numbers]).astype(np.float32))
                feature = np.concatenate(feature, axis=0)
                output_h5.create_dataset(i, data=feature.astype(np.float32))
       

if __name__ == "__main__":
    main()