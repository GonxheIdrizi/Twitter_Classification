#!/usr/bin/env python
# coding: utf-8

import numpy as np
import re
import string
import nltk
import csv
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
from pre_processing import *
import pandas as pd
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm


#-------------------------------------------------------------------------------------------------------------------

path = ''

#Read all the datasets for training and testing
with open(path+'./Datasets/train_neg.txt', 'r', encoding="utf8") as f:
    train_neg_sample = f.readlines()
with open(path+'./Datasets/train_pos.txt', 'r', encoding="utf8") as f:
    train_pos_sample = f.readlines()
with open(path+'./Datasets/test_data.txt', 'r', encoding="utf8") as f:
    data_test = f.readlines()
with open(path+'./Datasets/train_neg_full.txt', 'r', encoding="utf8") as f:
    train_neg_full = f.readlines()
with open(path+'./Datasets/train_pos_full.txt', 'r', encoding="utf8") as f:
    train_pos_full = f.readlines()

#Import the dictionnary of verb contractions
correction_dict = load_dict_contractions()


#-------------------------------------------------------------------------------------------------------------------
#Implement the intense cleaning of all positive and negative datasets, as well as the test dataset
print("intensively cleaning train_neg_sample")
intense_cleaned_neg_sample = intense_cleaning_tweets(train_neg_sample, correction_dict)
print("intensively cleaning train_pos_sample")
intense_cleaned_pos_sample = intense_cleaning_tweets(train_pos_sample, correction_dict)
print("intensively cleaning data_test")
intense_cleaned_data_test = intense_cleaning_tweets(data_test, correction_dict)
print("intensively cleaning train_neg_full: it will take a huge amount of time")
intense_cleaned_neg_full = intense_cleaning_tweets(train_neg_full, correction_dict)
print("intensively cleaning train_pos_full: it will take a huge amount of time")
intense_cleaned_pos_full = intense_cleaning_tweets(train_pos_full, correction_dict)

#Create the training intesively pre-processed datasets as well as their targets
train_all_sample_ic = np.concatenate((intense_cleaned_pos_sample, intense_cleaned_neg_sample), axis=0)
train_all_sample_ic_target = np.concatenate((np.ones(len(intense_cleaned_pos_sample)), np.zeros(len(intense_cleaned_neg_sample))),axis=0)
train_all_full_ic = np.concatenate((intense_cleaned_pos_full, intense_cleaned_neg_full), axis=0)
train_all_full_ic_target = np.concatenate((np.ones(len(intense_cleaned_pos_full)), np.zeros(len(intense_cleaned_neg_full))),axis=0)

#Create the csv files for our intensively cleaned datasets
path = './Datasets/'
print("Creating csv files")
create_csv_tweets(path,train_all_sample_ic,train_all_sample_ic_target, "train_sample_ic")
create_csv_tweets(path,train_all_full_ic,train_all_full_ic_target, "train_full_ic")
create_csv_tweets_test(path,intense_cleaned_data_test, "test_ic")
print("done.")

#-------------------------------------------------------------------------------------------------------------------
#Implement the softly cleaning of all positive and negative datasets, as well as the test dataset
print("softly cleaning train_neg_sample")
soft_cleaned_neg_sample = soft_cleaning_tweets(train_neg_sample, correction_dict)
print("softly cleaning train_pos_sample")
soft_cleaned_pos_sample = soft_cleaning_tweets(train_pos_sample, correction_dict)
print("softly cleaning data_test")
soft_cleaned_data_test = soft_cleaning_tweets(data_test, correction_dict)
print("softly cleaning train_neg_full")
soft_cleaned_neg_full = soft_cleaning_tweets(train_neg_full, correction_dict)
print("softly cleaning train_pos_full")
soft_cleaned_pos_full = soft_cleaning_tweets(train_pos_full, correction_dict)

#Create the training softly pre-rpocessed datasets as well as their targets
train_all_sample_sc = np.concatenate((soft_cleaned_pos_sample, soft_cleaned_neg_sample), axis=0)
train_all_sample_sc_target = np.concatenate((np.ones(len(soft_cleaned_pos_sample)), np.zeros(len(soft_cleaned_neg_sample))),axis=0)
train_all_full_sc = np.concatenate((soft_cleaned_pos_full, soft_cleaned_neg_full), axis=0)
train_all_full_sc_target = np.concatenate((np.ones(len(soft_cleaned_pos_full)), np.zeros(len(soft_cleaned_neg_full))),axis=0)

#Create the csv files for our softly cleaned datasets
path = './Datasets/'
print("Creating csv files")
create_csv_tweets(path,train_all_sample_sc,train_all_sample_sc_target, "train_sample_sc")
create_csv_tweets(path,train_all_full_sc,train_all_full_sc_target ,"train_full_sc")
create_csv_tweets_test(path,soft_cleaned_data_test, "test_sc")
print("done")


#-------------------------------------------------------------------------------------------------------------------

#Create the training datasets as well as their targets (not cleaned)
train_all_sample = np.concatenate((no_cleaning(train_pos_sample), no_cleaning(train_neg_sample)), axis=0)
train_all_sample_target = np.concatenate((np.ones(len(train_pos_sample)), np.zeros(len(train_neg_sample))),axis=0)
train_all_full = np.concatenate((no_cleaning(train_pos_full), no_cleaning(train_neg_full)), axis=0)
train_all_full_target = np.concatenate((np.ones(len(train_pos_full)), np.zeros(len(train_neg_full))),axis=0)

#Create the csv files for our not cleaned datasets
print("Creating csv files")
path = './Datasets/'
create_csv_tweets(path,train_all_sample,train_all_sample_target, "train_all_sample")
create_csv_tweets(path,train_all_full,train_all_full_target, "train_all_full")
create_csv_tweets_test(path,data_test, "test")

#Add some pre-processing because some characters cannot be read correctly
#Rectifications to data
with open('./Datasets/' + 'train_full_ic.csv', 'r', encoding='utf8', errors='ignore') as f:
    train_all_full_ic = f.readlines()
train_all_full_ic, train_all_full_ic_target = read_clean_data(train_all_full_ic)
create_csv_tweets('./Datasets/',train_all_full_ic,train_all_full_ic_target, "train_full_ic")

#Rectifications to data
with open('./Datasets/' + 'train_full_sc.csv', 'r', encoding='utf8', errors='ignore') as f:
    train_all_full_sc = f.readlines()
train_all_full_sc, train_all_full_sc_target = read_clean_data(train_all_full_sc)
create_csv_tweets('./Datasets/',train_all_full_sc,train_all_full_sc_target, "train_full_sc")

#Rectifications to data
with open('./Datasets/' + 'train_sample_sc.csv', 'r', encoding='utf8', errors='ignore') as f:
    train_all_sample_sc = f.readlines()
train_all_sample_sc, train_all_sample_sc_target = read_clean_data(train_all_sample_sc)
create_csv_tweets('./Datasets/',train_all_sample_sc,train_all_sample_sc_target, "train_sample_sc")

#Rectifications to data
with open('./Datasets/' + 'train_sample_ic.csv', 'r', encoding='utf8', errors='ignore') as f:
    train_all_sample_ic = f.readlines()
train_all_sample_ic, train_all_sample_ic_target = read_clean_data(train_all_sample_ic)
create_csv_tweets('./Datasets/',train_all_sample_ic,train_all_sample_ic_target, "train_sample_ic")

print("done")