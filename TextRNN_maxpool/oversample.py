import sys, os, re, csv, codecs, numpy as np, pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


def label_shuffle(X_train, Y_train):
    num_classes = len(Y_train[0]) + 1
    sampleX_list = [[] for i in range(num_classes)]
    sampleY_list = [[] for i in range(num_classes)]
    for i in range(len(Y_train)):
        count = 0
        for j in range(len(Y_train[0])):
            if (Y_train[i][j] == 1):
                count += 1
                sampleX_list[j].append(X_train[i])
                sampleY_list[j].append(Y_train[i])

        if (count == 0):
            sampleX_list[num_classes-1].append(X_train[i])
            sampleY_list[num_classes-1].append(Y_train[i])
    
    maxlen = 0
    labels_num = []
    for i in range(len(sampleX_list)):
        maxlen = max(maxlen, len(sampleX_list[i]))
        labels_num.append(len(sampleX_list[i]))
    print("the every label num is {}".format("\t".join(map(str,labels_num))))
    sampleX_list = np.asarray(sampleX_list)
    sampleY_list = np.asarray(sampleY_list)
    resultX_list = []
    resultY_list = []
    for i in range(num_classes):
        random_temp = np.arange(maxlen)
        np.random.shuffle(random_temp)
        random_temp = random_temp % len(sampleX_list[i])

        resultX_list.extend(np.asarray(sampleX_list[i])[random_temp])
        resultY_list.extend(np.asarray(sampleY_list[i])[random_temp])

    resultX_list = np.asarray(resultX_list)
    resultY_list = np.asarray(resultY_list)
    result_num = resultX_list.shape[0]
    result_index = np.arange(result_num)
    np.random.shuffle(result_index)
    resultX_list = resultX_list[result_index]
    resultY_list = resultY_list[result_index]
    print("the shape of resultX_list is {}".format(resultX_list.shape))
    print("the shape of resultY_list is {}".format(resultY_list.shape))
    return resultX_list, resultY_list

def label_shuffle_single(X_train, Y_train, num_classes):
    sampleX_list = [[] for i in range(num_classes)]
    sampleY_list = [[] for i in range(num_classes)]
    for i in range(len(Y_train)):
        j = Y_train[i]
        sampleX_list[j].append(X_train[i])
        sampleY_list[j].append(Y_train[i])
    maxlen = 0
    labels_num = []
    for i in range(len(sampleX_list)):
        maxlen = max(maxlen, len(sampleX_list[i]))
        labels_num.append(len(sampleX_list[i]))
    print("the every label num is {}".format("\t".join(map(str,labels_num))))
    sampleX_list = np.asarray(sampleX_list)
    sampleY_list = np.asarray(sampleY_list)
    resultX_list = []
    resultY_list = []
    for i in range(num_classes):
        random_temp = np.arange(maxlen)
        np.random.shuffle(random_temp)
        random_temp = random_temp % len(sampleX_list[i])

        resultX_list.extend(np.asarray(sampleX_list[i])[random_temp])
        resultY_list.extend(np.asarray(sampleY_list[i])[random_temp])

    resultX_list = np.asarray(resultX_list)
    resultY_list = np.asarray(resultY_list)
    np.random.shuffle(resultX_list)
    np.random.shuffle(resultY_list)
    print("the shape of resultX_list is {}".format(resultX_list.shape))
    print("the shape of resultY_list is {}".format(resultY_list.shape))
    return resultX_list, resultY_list


def test(filepath):
    with open(filepath,'rb') as file_in:
        t_x, t_y, v_x, v_y, embed_matrix = pickle.load(file_in)
    t_x, t_y = label_shuffle(t_x, t_y)
    with open("./file_oversample",'wb') as file_out:
        total = [t_x, t_y, v_x, v_y, embed_matrix]
        pickle.dump(total, file_out)
def test2():
    x=["the apple is good","the apple is great","the apple is bad"]
    y=[1, 1, 0]
    x, y = label_shuffle_single(x,y,2)
    print(x)
    print(y)
if __name__ == "__main__":
    #test("/home/lanchong/text-classification/data_process/file_train")
    test2()
    










