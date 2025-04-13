""" create data samples """
import logging
from collections import defaultdict

import lmdb
import math
import numpy as np
import pyarrow
import tqdm
from sklearn.preprocessing import normalize
import glob
import os

def calculate_mean_dir_vector(data_dir):
    """Calculate mean direction vector from all files in data_dir"""
    all_vectors = []
    
    for motion_file in glob.glob(os.path.join(data_dir, "*.npz")):
        motion_data = np.load(motion_file)
        dir_vectors = motion_data['direction_vectors']
        
        # Flatten direction vectors to match DiffGesture format
        vectors = dir_vectors.reshape(dir_vectors.shape[0], -1)
        all_vectors.append(vectors)
    
    # Concatenate all vectors and calculate mean
    all_vectors = np.concatenate(all_vectors, axis=0)
    mean_vector = np.mean(all_vectors, axis=0)
    
    return mean_vector

if __name__ == "__main__":
    data_dir = "/home/bsd/cospeech/DiffGesture/data/trinity/allRec"
    mean_vector = calculate_mean_dir_vector(data_dir)
    print(mean_vector)