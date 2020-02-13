# -*- coding: utf-8 -*-
from proj2_helpers import create_csv_submission
import pandas as pd


pred1= pd.read_csv('./Models/lstm_simplefc_full.csv')
pred2 = pd.read_csv('./Models/lstm_full.csv')
pred3 = pd.read_csv('./Models/cnn_glove_tweet_300dim.csv')


result = pd.concat([pred1,pred2,pred3], axis=1)
result = result.drop(columns='Id')

#We choose the majority of our 4 predictions model 
result = result.mode(axis=1)

predicted = result.iloc[:,0]
idx_test = [*range(1,len(predicted)+2)]

create_csv_submission(idx_test,predicted,'./Models/merging_lstm_and_cnn.csv')

