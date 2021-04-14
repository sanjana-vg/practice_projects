import numpy as np
import tensorflow as tf
import os
from os import scandir

'''
Reads the data from files present in the given path
If a single file contains multiple passages, reads each line of a file as a separate entry
Folder structure assumed:
    path
        - category1
            - file1
            - file2
        - category2
            - file1
            - file2
params
    path - base path for the data
    categories - an array of folder names (category names) present in the base path
    num_files - number of files to be read from each category
returns
    a map with category as key and a list of sentences as value
'''
def readTextFromFiles(path, categories, num_files):
    data = {}
    for category in categories:
        X = []
        file_path = path + category + '/'
        i = 0
        files = os.listdir(file_path)
        for file in files:
            if i > num_files:
                break
            i+=1
            with open(file_path + '/'+file, 'r') as datafile:
                text = datafile.readlines()
                X.extend(text[1:])
        data[category] = X
    return data


def test_readTextFromFiles():
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    data = readTextFromFiles('datasets/bbc-fulltext/bbc/', categories, 5)
    #print(data)
    for category in data:
        print(category, " - ", len(data[category]))


'''
Removes special characters and splits the data into train and test sets
params
    data - a map with category as key and a list of sentences as value
    train_test_split - percentage of data to be used for training set
returns
    X Train, Y train, X test, Y test
'''
def process_data(data, train_test_split):
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []
    for category in data:
        passages = data[category]
        train_size = len(passages) * train_test_split
        i = 0
        for passage in passages:
            p = passage.lower().strip().replace('.','').replace(',','').replace('\"', '')
            if(len(p) > 0):
                if i < train_size:
                    X_tr.append(p)
                    Y_tr.append(category)
                else:
                    X_te.append(p)
                    Y_te.append(category)
            i+=1
    return X_tr, Y_tr, X_te, Y_te

def test_process_data():
    train_test_split = 0.85
    X_train, Y_train, X_test, Y_test = process_data(data, train_test_split)
    print("training set size", len(X_train), len(Y_train))
    print("test set size", len(X_test), len(Y_test))

'''
Reads word embeddings for words in the vocabulary
params
    path - the file containing embeddings
returns
    embeddings - maps word to its embedding
    word_to_ind - maps each word to an index
    ind_to_word - maps each index to a word (reverse of word_to_ind)
    word_list - set of words present in the vocabulary
'''
def read_word_embeddings(path):
    embeddings = {}
    word_list = []
    with open(path, 'r', encoding = 'UTF-8') as file:
        line = file.readline()
        while line:
            emb = line.strip().split()
            embeddings[emb[0]] = np.array(emb[1:], dtype=np.float64)
            line = file.readline()
            word_list.append(emb[0])
        i = 1
        word_to_ind = {}
        ind_to_word = {}
        for w in sorted(embeddings):
            word_to_ind[w] = i
            ind_to_word[i] = w
            i = i + 1
    return embeddings, word_to_ind, ind_to_word, set(word_list)


'''
Splits input data into minibatches for training to take advantage of vectorization during gradient descent
params
    X_train, Y_train - dataset
    minibatch_size - number of inputs to be included in each mini batch
    seed - for random permutations
returns
    minibatches
'''
def split_into_mini_batches (X_train, Y_train, minibatch_size, seed = 1):
    np.random.seed(seed)
    m = X_train.shape[2]
    perm = np.random.permutation(m)
    X = X_train[:,:,perm]
    Y = Y_train[:,perm]
    
    mini_batches = []
    full = m // minibatch_size
    for i in range(0, full):
        mini_batches.append((X[:,:,(i*minibatch_size) : ((i+1)*minibatch_size)], Y[:, (i*minibatch_size) : ((i+1)*minibatch_size)]))
    if (i+1)*minibatch_size < m:
        X_temp = np.zeros((X_train.shape[0], X_train.shape[1], minibatch_size))
        Y_temp = np.zeros((Y_train.shape[0], minibatch_size))
        X_temp[:,:,0:m-((i+1)*minibatch_size)] = X[:,:,((i+1)*minibatch_size):m]
        Y_temp[:,0:m-((i+1)*minibatch_size)] = Y[:,((i+1)*minibatch_size):m]
        mini_batches.append((X_temp, Y_temp))
    return mini_batches

def test_split_into_mini_batches():
    X_test_mini = np.random.rand(2,3,5)
    Y_test_mini = np.random.randn(3,5)
    mini_batch_res = split_into_mini_batches (X_test_mini, Y_test_mini, 2, seed = 1)
    print(X_test_mini, Y_test_mini)
    print(mini_batch_res)
    
    
'''
Maps category to an index and vice versa
'''
def get_category_index_map(categories):
    category_to_index = {}
    index_to_category = {}
    for (i, category) in enumerate(categories):
        category_to_index[category] = i
        index_to_category[i] = category
    return category_to_index, index_to_category

