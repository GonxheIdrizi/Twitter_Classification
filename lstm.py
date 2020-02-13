# -*- coding: utf-8 -*-

from numpy import array
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras import backend as K
import pandas as pd 
from sklearn.model_selection import train_test_split
import nltk
import pickle
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.python.keras import Sequential
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
tf.keras.backend.clear_session()  

config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.Session(config=config_proto)
set_session(session)

SEED = 1234
nltk.download("popular")
train_all_csv= pd.read_csv('/content/gdrive/My Drive/ml_project2/Datasets/train_full_all.csv')

train_all, test_all = train_test_split(train_all_csv, test_size=0.2)

X_train = train_all['text']
y_train = train_all['label']
X_test = test_all['text']
y_test = test_all['label']

#Find the vocab size
all_words = []
for tweet in X_train:
  for word in tweet.split():
      all_words.append(word)

unique_words = list(set(all_words))
print(len(unique_words))

#tokenize
tokenizer = Tokenizer(num_words = len(unique_words)+10)
tokenizer.fit_on_texts(X_train)

#padding
X_train = tokenizer.texts_to_sequences(X_train)
word_count = lambda sentence: len(sentence)
longest_sentence = max(X_train, key=word_count)
pad = len(longest_sentence)
padded_sentences = pad_sequences(X_train, len(longest_sentence), padding='post')

#Save tokenizer
with open('./Datasets/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LSTMmodel(input_length, unique_words):

  # Build the model 
  embedding_vector_length = 200
  vocab_length = len(unique_words)+10
  model = Sequential() 
  model.add(Embedding(vocab_length+1, embedding_vector_length, input_length=input_length, mask_zero=True)) 
  model.add(LSTM(512)) 
  model.add(Dropout(0.4))
  model.add(Dense(212, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid')) 
  model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
  print(model.summary()) 
  return model, vocab_length+1

def LSTMmodel1(input_length, unique_words):
  # Build the model 
  embedding_vector_length = 200
  vocab_length = len(unique_words)+10
  model = Sequential() 
  model.add(Embedding(vocab_length+1, embedding_vector_length, input_length=input_length, mask_zero=True)) 
  model.add(LSTM(512)) 
  model.add(Dropout(0.4))
  model.add(Dense(1, activation='sigmoid')) 
  model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
  print(model.summary()) 
  return model, vocab_length+1

checkpoint = ModelCheckpoint('./Models/lstm2dense_1000_b64_drop0_2_full_checkpoint.hdf5', save_best_only=True, monitor='val_acc',mode='max')
model,vocab_length = LSTMmodel(padded_sentences.shape[1], unique_words)
model.fit(padded_sentences, y_train,batch_size=64, validation_split=0.1, epochs=2, verbose=1, callbacks = [checkpoint])

#PREDICTION
test_sample = pd.read_csv('./Datasets/test.csv')
x_preds = test_sample['text']
#tokenize
x_preds = tokenizer.texts_to_sequences(x_preds)
#padding
pad_x_preds = pad_sequences(x_preds, pad, padding='post')
print(pad_x_preds.shape)
y_pred = model.predict(pad_x_preds)

def reformat(pred_test):
    """
    reformat the model output to fit our submission format 
    (list of -1 or 1)
    """
    idx_test = list(range(1,len(pred_test)+1))
    predicted = []
    for pred in pred_test:
        if pred <0.5:
            predicted.append(-1)
        else:
            predicted.append(1)
    return idx_test, predicted

idx_test, predicted = reformat(y_pred.squeeze())

import csv
with open('./Datasets/lstm_full.csv', 'w+', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(idx_test, predicted):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})