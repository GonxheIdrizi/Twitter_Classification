# -*- coding: utf-8 -*-

from numpy import array
import numpy as np
import csv
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras import backend as K
import pandas as pd 
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from proj2_helpers import *

from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.python.keras import Sequential
import tensorflow as tf

#-------------------------------------------------------------------------------------------------------------------
# TOKENIZER CREATED BASED ON THE FULL TRAIN DATASET

# loading 2fc : tokeniser.pickle simplefc: tokenizer_full1.pickle
with open('./Datasets/tokenizer_full1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#-------------------------------------------------------------------------------------------------------------------
#LOADING + PRE-PROCESS OF THE TEST DATA
print("Loading the test data...")
test_sample = pd.read_csv('./Datasets/test.csv')
print("Test data loaded")
x_preds = test_sample['text']
#tokenize
x_preds = tokenizer.texts_to_sequences(x_preds)
#padding
pad = 68 #previously computed based on the x_train used for the model training
num_words = 515725 #  previously computed based on the x_train used for the model training : 2fc:514623, simplefc:515725
pad_x_preds = pad_sequences(x_preds, pad, padding='post')
print("Pre-processed ok.")

#-------------------------------------------------------------------------------------------------------------------
#BUILDING THE MODEL
print("Building of the model with embedding of 200")
embedding_vector_length = 200
model = Sequential() 
model.add(Embedding(num_words + 1, embedding_vector_length, input_length=pad_x_preds.shape[1], mask_zero=True)) 
model.add(LSTM(512)) 
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
model.summary()
print("Loading weights")
model.load_weights('./Models/lstm_full1_checkpoint.hdf5')
print("Weights loaded")

#-------------------------------------------------------------------------------------------------------------------
#PREDICTION
print("Start prediction")
y_pred = model.predict(pad_x_preds)
idx_test, predicted = reformat(y_pred.squeeze())
print("Prediction done")

submission_path = './Models/lstm_simplefc_full.csv'
with open(submission_path, 'w+', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(idx_test, predicted):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

print("Prediction save in {path}".format(path=submission_path))