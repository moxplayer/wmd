# the original file by Kusner et al. has been altered to fit the needs of the sentiment-analysis project
# 2018-11-09
from __future__ import division
import pdb, sys, numpy as np, pickle
from multiprocessing import Pool


import os
from datetime import datetime
from collections import namedtuple
from math import sqrt

sys.path.append('../util')
from utilities import cosine_distance, euclidean_distance

sys.path.append('./python-emd-master')
from emd import emd

# calculates the similarity (WMD) between two pickle files
# requires: [comparisonpair]: [(word_vectors1, word_weights1), (word_vectors1, word_weights1)]
#           [use_cosine]: bool: True if cosine distance should be used
# in order to apply the multiprocessing library, I removed the 'use_cosine' argument
def calc_similarity(comparisonpair):#, use_cosine):
    # load pickle files X, BOW_X = (word_vector_arrays, BOW-features)
    word_vectors1, word_weights1 = comparisonpair[0]
    word_vectors2, word_weights2 = comparisonpair[1]

    # for TESTING ONLY#######################################################
    #N = 100                                                               #
    #########################################################################
    # FOR TESTING ONLY!!!!!#################
    #X, BOW_X  = slice_it(X, BOW_X, N) #####
    #print(BOW_X)
    ########################################

    # check if both files users are identical
    if (word_vectors1 == word_vectors2) and (word_weights1 == word_weights2):
        return 1.0
    # else
    else:
        # calculate the earth mover's distance (EMD) between two 'signatures' (generalized distributions)
        # signature format: (list of vectors [number of vectors x embedding dimension], list of their weights)
        # with the cosine distance
        #if(use_cosine):
        emd_result = emd( (word_vectors1, word_weights1), (word_vectors2, word_weights2), cosine_distance)
        # map the EMD output to [0,1]:
        similarity = float(float(1)-(emd_result/2 * 1.0))   
        # or with the euclidean distance HERE you might have to distinguish between normalized and non-normalized
        #else:
        #    #print("Use euclidean-distance")
        #    emd_result = emd((word_vectors1, word_weights1), (word_vectors2, word_weights2), euclidean_distance)
        #    similarity = emd_result
    
        return similarity

# load a pickle file from a pickle_path
def load_pickle(pickle_path):
    # load both users' pickle files
    with open(pickle_path, 'rb') as f:
        pickle_load = pickle.load(f)
    return pickle_load

# export is a (Gephi) edge graph
# requires : [pickle_dir] folder with the corresponding pickle files to compair pairwise: if from_dir == True// else it is a list of paths to pickle files
def calcPairwiseDist(pickle_dir, similarity_dir, from_dir = True, experiment_name = ""):
    # use cosine
    use_cosine = True

    # export as gephi edge graph file
    as_gephi = True

    # collect pickle files to use:
    # if from dir:
    if from_dir == True:
        # 0. find all files in 'pickle_dir' (this works recursively - and all subfolders are searched)
        pickle_paths   = []
        for root, dirs, files in os.walk(pickle_dir):
            for name in files:
                if os.path.splitext(name)[1] == '.pk':
                    pickle_paths.append(os.path.join(root,name))
    # else: pickle_dir
    else:
        pickle_paths = pickle_dir
        
    

    #print('pickle_paths', pickle_paths)

    # 1. load comparison files
    comparison_files = [load_pickle(pickle_path) for pickle_path in pickle_paths]

    # 2. Generate comparison pairs and pickle path pairs
    comparison_pairs = []
    N                = len(pickle_paths)
    for i in xrange(N):
        for j in xrange(i+1,N):
            comparison_pairs.append((comparison_files[i], comparison_files[j]))
    # 3.Generate pickle path pairs (as node IDs)
    picklepath_pairs = []
    N                = len(pickle_paths)
    for i in xrange(N):
        for j in xrange(i+1,N):
            picklepath_pairs.append((pickle_paths[i], pickle_paths[j]))

    # NOTE: If you want to parallelize this, you will have to add the fileIDs to the similarity output to the 'calc_similarity' method [fuse picklepath_pairs and comparison_pairs]
    # 4. For all comparison pairs run emd (Earth Mover's Distance)
    similarities = []

    pool = Pool(processes=4)
    num_tasks = len(comparison_pairs)
    for  i, sim in enumerate(pool.map(calc_similarity, comparison_pairs), 1):
        sys.stderr.write('\rCalculated {}/{}({})% of all similarities'.format(i,num_tasks,round(i/num_tasks*100),2)) #0.%
        similarities.append(sim)
    print('')
    # 5. Convert to gephi edge-graph format
    # source, target, weight
    # ID1, ID2, weight1-2
    # ID1, ID3, weight1-3
    # ...
    if as_gephi:
        result = "source,target,weight\n"
        for picklepath_pair, similarity in zip(picklepath_pairs, similarities):
            filename1 = os.path.splitext(os.path.basename(str(picklepath_pair[0])))[0]
            filename2 = os.path.splitext(os.path.basename(str(picklepath_pair[1])))[0]
            
            line   = filename1 + ',' + filename2 + ',' + str(similarity)+'\n'
            result += line
    else:
        result = str(list(zip(picklepath_pairs, similarities)))[1:-1]

    # define a timesting (hour_minute_second) in order to specify the time the sim_file has been generated, NOT WORKING CURRENTLY
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_file_name = similarity_dir + time_string + '_' +experiment_name + '_sims.csv'
    with open(out_file_name, mode = 'w') as f:
        f.write(result)
        print('Wrote a file with pairwise similarities')
    return out_file_name

def main():
    pickle_dir     = sys.argv[1]
    similarity_dir = '../../data/similarityfiles/'

    # calculate all pairwise distances between pickle files (prepared by get_word_vectors.py --> should probably be renamed)
    calcPairwiseDist(pickle_dir, similarity_dir)


if __name__ == "__main__":
    main()


