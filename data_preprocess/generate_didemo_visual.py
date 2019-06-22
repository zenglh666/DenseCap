import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math
from sklearn.decomposition import PCA

input_file = '/home/zenglh/didemo/average_fc7.h5'
input_file_flow = '/home/zenglh/didemo/average_global_flow.h5'
output_file = '/home/zenglh/didemo/didemo_visual_pca.hdf5'


def main():
    features = []
    with h5py.File(input_file,'r') as input_h5:
        with h5py.File(input_file_flow,'r') as input_flow_h5:
            for k in input_h5:
                features.append(np.concatenate([input_h5[k], input_flow_h5[k]], axis=1).astype(np.float32))
    features = np.concatenate(features, axis=0)
    transformer = PCA(n_components=500)
    transformer.fit(features)

    with h5py.File(input_file,'r') as input_h5:
        with h5py.File(input_file_flow,'r') as input_flow_h5:
            with h5py.File(output_file,'w') as output_h5:
                keys = input_h5.keys()
                for k in keys:
                    feature = np.concatenate([input_h5[k], input_flow_h5[k]], axis=1)
                    output_h5.create_dataset(k, data=transformer.transform(feature).astype(np.float32))
            

if __name__ == "__main__":
    main()