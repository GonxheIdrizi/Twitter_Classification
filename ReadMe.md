Datasets, Glove Creation and Models folders can be loaded from the following google drive: https://drive.google.com/drive/folders/1ZB0KjTE_5h_jXKOBRD9y9Ryor7DOZ3tc?usp=sharing
Please put the Datasets, Glove Creation and Models folders in the Cleaned folder.

# Datasets folder
This folder contains all of our initial datasets, as well as the pre-processed one. Indeed, the intensive pre-processing takes a lot of time, so it was a good idea to also upload these csv files.
All the datasets used also for the embedding constructions are stored in this folder.

# pre_processing.py
Contains every useful method that helped us clean/pre-process our datasets, such as lemmatization, nouns removal, regex cleaning, a dictionary of verb and word contractions... It also contains methods to create our csv files.

# Data_Cleaning.py
Contains the pre-processing of all our datasets. It will create csv files with two columns, on for the text and one for the labels.
This file uses methods implemented in pre_processing.py.
All the pre-processed datasets are stored in the Datasets folder. They already are there as the intensive cleaning takes a long time. One can still run them if they want using this file.

# GloveCreation folder
This folder contains all functions used to create the glove embedding in 200 dimensions which is used in our models.
    -glove_helpers.py : contains all methods that allows one to create vectorized tweets.
    -glove_solution.py : methods that load cooccurrence matrix and use it to create the embedding of 200 dimensions.
    -features_creation.py : creates the embeddings per each datasets.
These embeddings will be stored in the Datasets folder.

# BasicModels.py
We have evaluated Naive Bayes, SVM and Linear Regression algorithms on multiple ways to handle the embedding (Glove and the sklearn.feature_extraction.text API)
The dataset we are using can be found in Datasets : test_data.txt, train_neg.txt, train_pos.txt for the one without pre-processing.
The submission of the test predictions are then stored in the Models folder.
If one wants to run on other datasets, he or she can find the different pre-processed datasets in the Datasets folder.

In the meantime, we used the CountVectorizer() provided by the sklearn.feature_extraction API to transform our text collection in a matrix of tokens. We then transformed the count matrix to a normalized tf-idf (term frequency inverse document-frequency) representation.

In both cases, we used the l2-norm to compare our vectors. We are using GridSearch to try multiple hyperparameters and cross validation with 5 folds (this one is the one that gives us the best results).

# BasicModelsWithGloVe.py
We choose to train the models Naive Bayes, SVM and Linear Regression with our own embedding based on the tweet dataset that was provided. We used the GloVe embedding. We have tried different embeddings sizes (20,100,200,300) and best results were obtained with 200 dimensions.
The submission of the test predictions are then stored in the Models folder.

# bi-gram_Model.py -Sentiment analysis
A key concept in FastText is to compute the n-grams of an input sentence and append them to the end of a sentence. In our implementation, we'll use bi-grams. Briefly, a bi-gram is a pair of words/tokens that appear consecutively within a sentence.
Example: "I love Machine Learning" gives the following bi-grams: ["I love", "love Machine", "Machine Learning"]

The criterion we use is the BCEWithLogitsLoss that combines a Sigmoid layer and the BCELoss in a single class.

- "train_full" and "test" datasets are csv files with each 2 columns : 'label' and 'text'
- We used pre-trained GloVE embedding from a twitter dataset of 27 billion of tweets with 200 dimensions

- We have constructed several neural networks, our best one is:
    input -> Embedding layer -> average pool function -> Linear layer -> average pool func -> Linear layer -> output
We chose not to go further with the FastText2 as it was giving us really bad predictions. Thus we used only FastText1.
- We defined the Adam optimizer, a BCEWithLogitsLoss as our criterion, trained our model on the training datasets and evaluated our model on the validation dataset.
- We stopped at the correct epoch before overfitting (the epoch parameter is at 5)
- We predicted our test data and saved it for submission in the Models folder.


# cnn.py -Sentiment analysis
We write down some details of the implementation of our convolutional neural network:
- We splited our training datasets into a training set and a validation set.
- We used max pooling (to extract more important features) and this function handles sequences with different lengths which is important for our convolution filter since their outputs are depending on the input size.
- We use dropout on our convolution filter + a linear function at the end
- We use the same optimizer, criterion, training and validating methods as in the bi_gram_Model.py.
- We have constructed two different neural networks, both with the same convolution filters, meaning 2,3,4 and 5 and our best one was:
 input -> Embedding layer -> Convolutions Layers -> ReLU activation fonction -> pooled1d activation function for each output of the preceding layer + concatenation -> Dropout -> Linear layer -> Output
The second CNN gave us poor prediction, so we have chosen to not use it for the result description.
- For the prediction, we need to make sure that the input is at least as long as the largest filter we are using. So don't forget to change the min_len accordingly.

The convolution filters act as our n-gram with n=2,3... of the fast text but we don't need to figure out now which one is the most interesting because we are testing a lot of them among layers. Theses filters are contained in the convs of each model.

# ModelsBagging.py
A way of combining predictions from multiple models. We combine them such that for each input (test tweet), we take the label which appears in majority (1 or -1).

# lstm.py
This is the implementation for the Long-short-term memory (LSTM) model
We first loaded our training dataset, tokenized it according to the nltk library. Then, we splited it into a train and
validation set, set the vocabulary and constructed the embedding based on the vocabulary in the training set.
We constructed our model as :
  input -> Embedding layer -> LSTM with 512 units -> Linear function with the sigmoid activation function.
This model gives us our best accuracy when it was trained on the full dataset.

# run.py

This folder will enable you to run our best model (LSTM). You have to first install keras.
If you are working with anaconda you can do it as follow:
- conda install -c conda-forge keras
Otherwise, keras is installed by default on Google Colab.

This contains a simplified version of the lstm.py where the model has already been trained:
1) You load your tokenizer : this one was saved during the training part as it is based on the training vocabulary
2) You load and preprocess accordingly your test data: for this part, you need the same tokenizer as the training.
We saved it but 2 more things were computed during the training and cannot be find back : the padding (that is the maximum length of a tweet in the training set) and the num_words (that is the number of unique words found in the training
set). We have put them directly on the file for you, otherwise you should run the lstm.py and train again your model. Take care of manually changing these values after training your model.
3) You have to load the weights precomputed during the training phase, you can find them in the Models folder.
4) At the end, you predict your lstm and save it in Models as lstm_simplefc_full.csv.

# Models
This folder contains some of our prediction as well as our weights for the lstm model. 

