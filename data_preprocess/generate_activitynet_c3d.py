import numpy as np
import tensorflow as tf
import h5py
import json
import sys
import os 
import math

input_file = '/home/zenglh/activitynet/sub_activitynet_v1-3.c3d.hdf5'
output_file = '/home/zenglh/activitynet/activitynet_c3d.hdf5'


def main():
    with h5py.File(input_file,'r') as input_h5:
        with h5py.File(output_file,'w') as output_h5:
            keys = input_h5.keys()
            for k in keys:
                output_h5.create_dataset(k, data=input_h5[k]["c3d_features"][:].astype(np.float32))
            

if __name__ == "__main__":
    main()