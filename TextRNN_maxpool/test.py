# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
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
import pandas as pd

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",6,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 200, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","text_rnn_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",800,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",50,"embedding size")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("file_path","../data_process/clean_data/data_file","train data and val data and embedding matrix")
#tf.app.flags.DEFINE_string("traning_data_path","train-zhihu4-only-title-all.txt","path of traning data.") #train-zhihu4-only-title-all.txt===>training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
#tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec.bin-100","word2vec's vocabulary and vectors")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)


def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    with open(FLAGS.file_path, 'rb') as data_f:
        X_t,train_y,embedding_matrix,X_te= pickle.load(data_f)
        vocab_size = len(embedding_matrix)
        #X_te = pad_sequences(X_te, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        #print("total train sample num is %s" %len(trainX))

    '''
    if 1==1:
        #1.  get vocabulary of X and label.
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',word2vec_model_path=FLAGS.word2vec_model_path,name_scope="rnn")
        vocab_size = len(vocabulary_word2index)
        print("rnn_model.vocab_size:",vocab_size)
        vocabulary_word2index_label,vocabulary_index2word_label = create_voabulary_label(name_scope="rnn")
        train, test, _ =  load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,multi_label_flag=False,traning_data_path=FLAGS.traning_data_path) #,traning_data_path=FLAGS.traning_data_path
        trainX, trainY = train
        testX, testY = test
        # 2.Data preprocessing.Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        ###############################################################################################
        #with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
        #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
        ###############################################################################################
        print("trainX[0]:", trainX[0]) #;print("trainY[0]:", trainY[0])
        # Converting labels to binary vectors
        print("end padding & transform to one hot...")
    '''
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sample_submission = pd.read_csv('../data_process/sample_submission.csv')
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textRNN=TextRNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
        vocab_size, FLAGS.embed_size, FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        result = predict(sess,textRNN,X_te,FLAGS.batch_size)

        sample_submission[list_classes] = result
        sample_submission.to_csv('submission.csv', index=False)

        '''
        curr_epoch=sess.run(textRNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                curr_loss,curr_acc,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.train_op],feed_dict={textRNN.input_x:trainX[start:end],textRNN.input_y:trainY[start:end]
                    ,textRNN.dropout_keep_prob:1}) #curr_acc--->TextCNN.accuracy -->,textRNN.dropout_keep_prob:1
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %1==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textRNN.epoch_increment)
            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss, eval_acc=do_eval(sess,textRNN,testX,testY,batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        #test_loss, test_acc = do_eval(sess, textRNN, testX, testY, batch_size,vocabulary_index2word_label)
        '''
    pass

'''
def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textRNN,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
'''

def assign_pretrained_word_embedding(sess, embed_matrix, textRNN):
    word_embedding = tf.constant(embed_matrix, dtype=tf.float32)
    t_assign_embedding = tf.assign(textRNN.Embedding, word_embedding)
    sess.run(t_assign_embedding)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
'''
def do_eval(sess,textRNN,evalX,evalY,batch_size,vocabulary_index2word_label):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, logits,curr_eval_acc= sess.run([textRNN.loss_val,textRNN.logits,textRNN.accuracy],#curr_eval_acc--->textCNN.accuracy
                                          feed_dict={textRNN.input_x: evalX[start:end],textRNN.input_y: evalY[start:end]
                                              ,textRNN.dropout_keep_prob:1})
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)
'''
def do_eval(sess,textRNN,evalX,evalY,batch_size):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, logits,curr_eval_acc= sess.run([textRNN.loss_val,textRNN.logits,textRNN.accuracy],#curr_eval_acc--->textCNN.accuracy
                                          feed_dict={textRNN.input_x: evalX[start:end],textRNN.input_y: evalY[start:end]
                                              ,textRNN.dropout_keep_prob:1})
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)
'''
def predict(sess, textRNN, testX,batch_size):
    number_examples=len(testX)
    result = np.zeros((number_examples, 6))
    curr = 0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        logits = sess.run(textRNN.probality,#curr_eval_acc--->textCNN.accuracy
                                          feed_dict={textRNN.input_x: testX[start:end]
                                              ,textRNN.dropout_keep_prob:1})
        #logits = tf.sigmoid(logits)
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        result[curr:curr+batch_size] = logits
        curr = curr+batch_size
    return result
'''
def predict(sess, textRNN, testX,batch_size):
    number_examples=len(testX)
    result = np.zeros((number_examples, 6))
    start = 0
    end = number_examples
    while(start < end):
        if start+batch_size < end:
                
            logits = sess.run(textRNN.probality,#curr_eval_acc--->textCNN.accuracy
                                            feed_dict={textRNN.input_x: testX[start:start+batch_size]
                                                ,textRNN.dropout_keep_prob:1})
            result[start:start+batch_size] = logits

        else:
            logits = sess.run(textRNN.probality,#curr_eval_acc--->textCNN.accuracy
                                            feed_dict={textRNN.input_x: testX[start:end]
                                                ,textRNN.dropout_keep_prob:1})  
            result[start:end] = logits          
        #logits = tf.sigmoid(logits)
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        
        start = start+batch_size 
    return result       


    '''
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        logits = sess.run(textRNN.probality,#curr_eval_acc--->textCNN.accuracy
                                          feed_dict={textRNN.input_x: testX[start:end]
                                              ,textRNN.dropout_keep_prob:1})
        #logits = tf.sigmoid(logits)
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        result[curr:curr+batch_size] = logits
        curr = curr+batch_size
    return result
    '''


#从logits中取出前五 get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):
    #print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

if __name__ == "__main__":
    tf.app.run()