import sys, os, re, csv, codecs, numpy as np, pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
np.random.seed(2018)


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_sentences_train = train["comment_text"].fillna("_na_").values
list_sentences_test = test["comment_text"].fillna("_na_").values
train_y = train[list_classes].values
#test_y = test[list_classes].values
corpus = list(list_sentences_test) + list(list_sentences_train) 
def get_len_string(total_tokenized):
    result = []
    for temp in total_tokenized:
        result.append(len(temp.split()))
    result = np.asarray(result)
    print(result.shape)
    print("mean:%.3f\tstd:%.3f"%(result.mean(), result.std()))
    print(np.median(result))
    return result.mean(), result.std()
get_len_string(corpus)

'''
EMBEDDING_FILE = "./glove.42B.300d.txt"
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(corpus))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)




X_t = pad_sequences(list_tokenized_train, maxlen=800)
X_te = pad_sequences(list_tokenized_test, maxlen=800)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

embed_size = 300
word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    #if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print(nb_words)
total = [X_t,train_y,embedding_matrix,X_te]


with open("data_file","wb") as file_out:
    pickle.dump(total, file_out)

with open("train_data", "wb") as file_out:
    pickle.dump(X_t, file_out)
with open("test_data", "wb") as file_out:
    pickle.dump(X_te, file_out)   
'''




