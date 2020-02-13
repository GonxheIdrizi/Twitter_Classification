import numpy as np
import re
import string
import csv
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
from pre_processing import *
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm


def load_dict_contractions():
    """
    Loads a dictionnary of verb contractions and rewrites them correctly
    """    
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "arent":"are not",
        "can't":"can not",
        "cant":"can not",
        "'cause":"because",
        "couldn't":"could not",
        "couldnt":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "don't":"do not",
        "dont":"do not",
        "doesn't":"does not",
        "doesnt":"does not",
        "didn't":"did not",
        "didnot":"did not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gimm":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "haven't":"have not",
        "hasn't":"has not",
        "hasnt":"has not",
        "hadnt":"had not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "i'd":"i would",
        "i'll":"i will",
        "i'm":"i am",
        "im":"i am",
        "i'm'a":"i am about to",
        "i'm'o":"i am going to",
        "isn't":"is not",
        "isnt":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "i've":"i have",
        "ive":"i have",
        "kinda":"kind of",
        "let's":"let us",
        "lets":"let us",
        "luv":"love",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldnt":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "sux":"sucks",
        "u":"you",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "wasnt":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "weve":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wont":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "yall":"you all",
        "you'd":"you would",
        "youd":"you would",
        "you'll":"you will",
        "youll":"you will",
        "you're":"you are",
        "youre":"you are",
        "you've":"you have",
        "youve":"you have",
        "whatcha":"What are you",
        "wasn't":"was not", 
        "wasnt":"was not",
        "weren't":"were not",
        "won't":"will not",
        "wont":"will not",
        "wouldn't":"would not", 
        }



#Intense pre-processing of the tweets: gives bad accuracy



#Returns a cleaned string
def intense_regex_tweets(tweet):
    """
    Cleans words of tweets, such as removing numbers, <user>, <url>, ...
    Arguments: tweet (a string)
    """
    #Lowers every caracters
    tweet = tweet.lower()
    #replace "<3" by love
    tweet = re.sub(r"<3", r"love", tweet)
    #For words like "loooovveeee", it will return "loovve" (keep at most duplicates)
    tweet = re.sub(r"(.)\1+", r"\1\1", tweet)
    #rewrite word "love" correctly
    tweet = re.sub(r"[l]+[o]*[v]+[e]*\w*", r"love", tweet)
    #replace "hahaha" word by "laugh"
    tweet = re.sub(r"a*ha*h[ha]*", r"laugh", tweet)
    #replace "lma(f)o" by "laugh"
    tweet = re.sub(r"[l]+[m]+[f]*[a]+[o]+", r"laugh", tweet)
    #replace "lmao" by "laugh"
    tweet = re.sub(r"[l]+[m]+[f]*[a]+[o]+", r"laugh", tweet)
    #replace "lol" by "laugh"
    tweet = re.sub(r"[l]+[o]+[l]+", r"laugh", tweet)
    #rewrite word "not" correctly
    tweet = re.sub(r"[n]+[o]*[t]+", r"not", tweet)
    #remove <user>
    tweet = re.sub(r"<user>", r"", tweet)
    #remove <url> 
    tweet = re.sub(r"<url>", r"", tweet)
    #remove digits
    tweet = re.sub(r"\d", r"", tweet)
    #remove single characters 
    tweet = re.sub(r'\b\w\b', '', tweet)
    return tweet

def remove_nouns(tweet):
    """
    Cleans one tweet by removing proper nouns from it
    Arguments: tweet (a string)
    """
    tagged_sentence = nltk.tag.pos_tag(tweet.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    tweet = ' '.join(edited_sentence)
    return tweet

def lemmatization(tweet_tokenized):
    """
    Lemmatizes one tweet by generating the root form of each word of the tweet
    Arguments: tweet_tokenized (a list of words)
    """
    lancaster = LancasterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer() 
    lines_with_lemmas_word = []
    lines_with_lemmas_lanc = []
    temp_line_word = []
    temp_line_lanc = []
    for word in tweet_tokenized.split(): 
        temp_line_word.append(wordnet_lemmatizer.lemmatize(word, pos='v'))
        temp_line_lanc.append(lancaster.stem(word))
    lines_with_lemmas_word.append(' '.join(temp_line_word))
    lines_with_lemmas_lanc.append(' '.join(temp_line_lanc))
    return (lines_with_lemmas_word, lines_with_lemmas_lanc)

def intense_clean_tweet(tweet,dic):
    """
    Cleans tweet by applying some regex cleaning, removing punctuations, removing proper nouns, correcting contractions of verbs,...
    Arguments: tweet (a string)
               dic (a dictionnary of contractions)
    """
    tokenizer = TweetTokenizer()
    #remove proper nouns
    tweet = remove_nouns(tweet)
    #clean using regex
    tweet = intense_regex_tweets(tweet)
    words = tokenizer.tokenize(tweet)
    cleaned = [dic[word] if word in dic.keys() else word for word in words]
    tweet = " ".join(cleaned)
    #remove punctuation
    tweet = ''.join(word.strip(string.punctuation) for word in tweet)
    words = []
    #remove english stopwords
    for word in tweet.split():
        if word not in stopwords.words('english'):
            words.append(word)
    tweet = ' '.join(word for word in words)
    #remove multiple white spaces
    tweet = re.sub(' {1,}', ' ', tweet)
    return tweet

def intense_cleaning_tweets(ls,dic):
    """
    Pre-processes a list of tweets
    Arguments: ls (a list of tweets, which are strings)
               dic (a dictionnary of contractions)
    """
    cleaned_tweets = []
    for tweet in tqdm(ls):
        tweet = intense_clean_tweet(tweet, dic)
        tweet = lemmatization(tweet)[0][0]
        cleaned_tweets.append(tweet)
    return cleaned_tweets



#Soft pre-processing of the tweets: gives good accuracy



#Returns a cleaned string
def soft_regex_tweets(tweet):
    """
    Cleans words of tweets, such as removing numbers, <user>, <url>, ...
    Arguments: tweet (a string)
    """
    #Lowers every caracters
    tweet = tweet.lower()
    #replace "<3" by love
    tweet = re.sub(r"<3", r"love", tweet)
    #For words like "loooovveeee", it will return "loovve" (keep at most duplicates)
    tweet = re.sub(r"(.)\1+", r"\1\1", tweet)
    #rewrite word "love" correctly
    tweet = re.sub(r"[l]+[o]*[v]+[e]*\w*", r"love", tweet)
    #replace "hahaha" word by "laugh"
    tweet = re.sub(r"a*ha*h[ha]*", r"laugh", tweet)
    #replace "lma(f)o" by "laugh"
    tweet = re.sub(r"[l]+[m]+[f]*[a]+[o]+", r"laugh", tweet)
    #replace "lmao" by "laugh"
    tweet = re.sub(r"[l]+[m]+[f]*[a]+[o]+", r"laugh", tweet)
    #replace "lol" by "laugh"
    tweet = re.sub(r"[l]+[o]+[l]+", r"laugh", tweet)
    #rewrite word "not" correctly
    tweet = re.sub(r"[n]+[o]*[t]+", r"not", tweet)
    #remove <user>
    tweet = re.sub(r"<user>", r"", tweet)
    #remove <url> 
    tweet = re.sub(r"<url>", r"", tweet)
    #remove digits
    tweet = re.sub(r"\d", r"", tweet)
    return tweet

def soft_clean_tweet(tweet,dic):
    """
    Cleans tweet by applying some regex cleaning, removing punctuations, removing proper nouns, correcting contractions of verbs,...
    Arguments: tweet (a string)
               dic (a dictionnary of contractions)
    """
    tokenizer = TweetTokenizer()
    #clean using regex
    tweet = soft_regex_tweets(tweet)
    words = tokenizer.tokenize(tweet)
    cleaned = [dic[word] if word in dic.keys() else word for word in words]
    tweet = " ".join(cleaned)
    #remove punctuation
    tweet = ''.join(word.strip(string.punctuation) for word in tweet)
    #remove multiple white spaces
    tweet = re.sub(' {1,}', ' ', tweet)
    return tweet

def soft_cleaning_tweets(ls,dic):
    """
    Pre-processes a list of tweets
    Arguments: ls (a list of tweets, which are strings)
               dic (a dictionnary of contractions)
    """
    cleaned_tweets = []
    for tweet in tqdm(ls):
        tweet = soft_clean_tweet(tweet, dic)
        cleaned_tweets.append(tweet)
    return cleaned_tweets

def no_cleaning(ls):
    """
    list of tweets
    Arguments: ls (a list of tweets, which are strings)
    """
    cleaned_tweets = []
    for tweet in ls:
        cleaned_tweets.append(tweet)
    return cleaned_tweets


#Other useful methods

def read_clean_data(train):
    """
    Generates the cleaned arrays
    Arguments: train (the training data)
    """
    train = np.array([t.split(',') for t in train])
    train[:,1] = [l.replace('\n', '') for l in train[:,1]]
    train[:,1] = [l.replace('.0', '') for l in train[:,1]]
    train = train[1:,:]
    train_all = train[:,0]
    train_all_target = [int(l) for l in train[:,1]]
    return (train_all, train_all_target)


def build_vocab(tweets):
    """
    builds a alphabetically sorted list of words from the pre-processed tweets
    Arguments: tweets (a list of pre-processed tweets)
    """
    words = []
    for t in tweets:
        for word in t.split():
            words.append(word)
    words = list(set(words))
    return sorted(words)

def create_csv_tweets(path,train_all,train_all_target,name):
    """
    Creates an output file in csv
    Arguments: tweets (an array of tweets)
               name (string name of .csv output file to be created)
    """
    name = path + name + ".csv"
    with open(name, 'w+' , newline='') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for text, label in tqdm(zip(train_all, train_all_target)):
            try:
                writer.writerow({'text':text,'label':label})
            except:
                print("")
                
def create_csv_tweets_test(path,test_sample,name):
    """
    Creates a test output file in csv
    Arguments: tweets (an array of test tweets)
               name (string name of .csv output file to be created)
    """
    name = path + name + ".csv"
    with open(name, 'w+' , newline='') as csvfile:
            fieldnames = ['text', 'label']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for text in tqdm(test_sample):
                try:
                    writer.writerow({'text':text,'label':None})
                except:
                    print(text)