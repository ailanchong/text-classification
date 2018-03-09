import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import pickle
import sys
import tensorflow as tf
import numpy as np
from p8_TextRNN_model import TextRNN
#from data_util_zhihu import load_data_multilabel_new,create_voabulary,create_voabulary_label
#from tflearn.data_utils import pad_sequences #to_categorical
import os
from keras.preprocessing.sequence import pad_sequences
#import word2vec
import pickle
import matplotlib.pyplot as plt
from oversample import label_shuffle
from sklearn.metrics import roc_auc_score
maxlen = 200
embed_size = 300

with open("/input/file_train", 'rb') as data_f:
    #orignal_X, orignal_Y, testX, testY, embed_matrix= pickle.load(data_f)
    trainX, trainY, testX, testY, embed_matrix= pickle.load(data_f)
    vocab_size = len(embed_matrix)
    print(vocab_size)
    #print("total train sample num is %s" %len(orignal_X))
    trainX = pad_sequences(trainX, maxlen=maxlen, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=maxlen, value=0.)  # padding to max length

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(embed_matrix), embed_size, weights=[embed_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x1 = Bidirectional(GRU(128, return_sequences=True))(x)
    #x2 = Bidirectional(GRU(64, return_sequences=True))(x)
    #conc = concatenate([x1, x2])
    conc = x1
    avg_pool = GlobalAveragePooling1D()(conc)
    max_pool = GlobalMaxPooling1D()(conc)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(64, activation='relu')(conc)
    x = Dropout(0.1)(x)
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

batch_size = 512
epochs = 10
model = get_model()
RocAuc = RocAucEvaluation(validation_data=(testX, testY), interval=1)
hist = model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY),
                 callbacks=[RocAuc], verbose=1)
model.save('/output/my_model.h5')


