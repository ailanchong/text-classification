import sys, os, re, csv, codecs, numpy as np, pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
np.random.seed(2018)

def shuffle_split(train_x, train_y, ratio=0.1):
    index = np.arange(len(train_x))
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    train_num = int(len(train_x)*(1-ratio))
    t_x = train_x[0:train_num]
    t_y = train_y[0:train_num]
    v_x = train_x[train_num:-1]
    v_y = train_y[train_num:-1]
    return t_x, t_y, v_x, v_y


with open("../data/twice/data_file",'rb') as file_in:
    trainX, trainY, embed_matrix,test_x = pickle.load(file_in)
    t_x, t_y, v_x, v_y = shuffle_split(trainX, trainY, ratio=0.1)

print("train sample num is %s" %len(t_x))

total = [t_x, t_y, v_x, v_y, embed_matrix]

with open("../data/twice/file_train",'wb') as file_in:
    pickle.dump(total, file_in)

