import sys, os, re, csv, codecs, numpy as np, pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from clean import clean
np.random.seed(2018)

def get_set(filepath):
    result = set()
    with open(filepath, 'r') as file_in:
        for line in file_in:
            result.add(line.strip())
    return result
def get_map(filepath):
    result = {}
    with open(filepath,'r') as file_in:
        for line in file_in:
            line = line.strip().split(",")
            result[line[0]] = line[1]
    return result
def init_datastructure(swearfile, corrfile, knowsetfile, keysetfile, commonsetfile):
    swear_words = get_set(swearfile)
    corr_map = get_map(corrfile)
    knowset = get_set(knowsetfile)
    keysetfile = get_set(keysetfile)
    commonsetfile = get_set(commonsetfile)
    return swear_words, corr_map, knowset, keysetfile, commonsetfile

def get_len_list(total_tokenized):
    result = []
    for temp in total_tokenized:
        result.append(len(temp))
    result = np.asarray(result)
    print(result.shape)
    print("mean:%.3f\tstd:%.3f"%(result.mean(), result.std()))
    print(np.median(result))
    return result.mean(), result.std()
def get_len_string(total_tokenized):
    result = []
    for temp in total_tokenized:
        result.append(len(temp.split()))
    result = np.asarray(result)
    print(result.shape)
    print("mean:%.3f\tstd:%.3f"%(result.mean(), result.std()))
    print(np.median(result))
    return result.mean(), result.std()

swear_words, corr_map, knowset, keyset, commonset = init_datastructure("../data/swear_words.csv","../data/correct_words.csv","../data/know_set",
                                                                              "../data/keyword_set","../data/common_set")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_sentences_train = train["comment_text"].fillna("_na_").apply(lambda x:clean(x,swear_words, corr_map, knowset, keyset, commonset))
list_sentences_test = test["comment_text"].fillna("_na_").apply(lambda x:clean(x,swear_words, corr_map, knowset, keyset, commonset))
train_y = train[list_classes].values
#test_y = test[list_classes].values
corpus = list(list_sentences_test) + list(list_sentences_train) 
get_len_string(corpus)
with open("../data/clean_corpus",'w') as file_out:
    for line in corpus:
        file_out.write(line + "\n")

'''
corpus = []
with open("./clean_corpus",'r') as file_in:
    for line in file_in:
        line = line.strip()
        corpus.append(line)
print(len(corpus))
'''
EMBEDDING_FILE = "../data/glove.42B.300d.txt"
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(list(corpus))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

total_tokenized = list_tokenized_test + list_tokenized_train


get_len_list(total_tokenized)


X_t = pad_sequences(list_tokenized_train, maxlen=800)
X_te = pad_sequences(list_tokenized_test, maxlen=800)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = 300
word_index = tokenizer.word_index
nb_words = len(word_index)+1
print(nb_words)
in_num = 0
out_num = 0
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
vocab_in = open("./vocab_in",'w')
vocab_out = open("./vocab_out",'w')
for word, i in word_index.items():
    #if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
        in_num += 1
        vocab_in.write(word + '\n')
    else:
        out_num += 1
        vocab_out.write(word + '\n')
vocab_in.close()
vocab_out.close()
print(nb_words)
print("in_num:%d\tout_num:%d"%(in_num, out_num))

total = [X_t,train_y,embedding_matrix,X_te]


with open("data_file","wb") as file_out:
    pickle.dump(total, file_out)

