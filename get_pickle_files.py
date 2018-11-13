# 2018-11-09
# this file has been altered to fit this sentiment-analysis project's needs
# read txt files in 'train_folder' which holds txt files with word tokens separated by blanks
# for every such file pickle files are generated
import gensim, pdb, sys, pickle, string, os
import scipy.io as io
import numpy as np
import fastText


# requires [word_list]: a list of word strings to be converted to vectors
#          [model]    : a fastText (wordembedding) model
def apply_model(word_list, model):
    # extract word counts and word vectors
    # word vectors to compars (words x embedding_size)
    word_vectors = np.array([model.get_word_vector(word) for word in word_list])

    # transpose  <word_vectors> and cast to list data type
    #transposed_wordvectors = word_vectors.T # [~np.all(word_vectors.T == 0, axis=1)]
    result                 = word_vectors.tolist()#transposed_wordvectors.tolist()
    return result

# create a pickle file for WMD comparison
# requires: transposed_wordvectors : np.array of transposed word embedding outputs
#           np_occurrences         : list of word_weights
#           pickle_name            : path for the output pickle file
def create_pickle(transposed_wordvectors, word_weights, pickle_path):
    # word weights have to be floats
    word_weights = [float(el) for el in word_weights]
    # pickle the <transposed_wordvectors> and a numpy array of occurrences (Bag of Words)
    pickle_load = [transposed_wordvectors, word_weights]
    # try pickling
    with open(pickle_path , 'wb') as f:
        pickle.dump(pickle_load, f)
        f.close()
    return

# read a list of space separated word tokens
def read_word_list(text_path):
    with open(text_path, mode='r') as f:
        text = f.read()
    word_list = text.split()



def main():
    # 0. specify a folder with trainign datasets (*.txt files)
    work_token_file = sys.argv[1] # e.g.: 'twitter.txt'
    we_model_path   = sys.argv[2] # e.g.: 'reference_model.bin'

    # 1. load a word embedding model (e.g. word2vec trained on Google News - 'GoogleNews-vectors-negative300.bin') in a binary format
    # load the reference model
    model    = fastText.load_model(we_model_path)

    # for all *.txt files:
    word_token_path = os.path.join(root,name)
    
    # 2. read word_list
    word_list = read_word_list(text_path)    

    # 3. apply the word embedding model to all words in the word list
    # and transpose afterwards
    transposed_wordvectors = apply_model(word_list, model)

    # 4. read/create word_weights
    word_weights = [1 for _ in word_list]

    # 5. create pickle file
    pickle_directory = "../../data/picklefiles/"
    pickle_name      = pickle_directory + os.path.splitext(os.path.basename(train_dataset))[0] + ".pk"
    create_pickle(transposed_wordvectors, word_weights, pickle_name)


if __name__ == "__main__":
    main()                                                                                            