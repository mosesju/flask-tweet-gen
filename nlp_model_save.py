import gensim
import numpy as np
import pandas as pd
import nltk
import os

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.utils.data_utils import get_file
# class TweetGen:
def main_control(phrase1, phrase2, phrase3):
    fin_list = []
    #gets tweets from file
    file_name = 'tweets_edited.txt'
    tweets = [line.rstrip('\n') for line in open(file_name,encoding='utf-8')]
    #gets words from tweets for gensim, then removes blank lists
    # Used for list comprehension
    tweet_words = [tweet.lower() for tweet in tweets]
    tweet_words = filter(None, tweet_words)
    tweet_words = [tweet.split() for tweet in tweet_words]
    # 
    main_control.phrae1 = phrase1
    main_control.phrae2 = phrase2
    main_control.phrae3 = phrase3

    #loads gensim model
    # model = gensim.models.Word2Vec.load('model.bin')
    print("Gensim")
    model = gensim.models.Word2Vec(tweet_words,min_count=2)
    model.save('model.bin')
    model.train(tweet_words,total_examples=model.corpus_count,epochs=10)
    print("Gensim model Trained")
    main_control.model = model
    weights = model.wv.vectors
    #Basic data about dataset
    word_count = lambda sentence: len(sentence)
    max_sent_len = len(max(tweets,key=word_count))
    main_control.max_sent_len = max_sent_len
    vocab_size = len(model.wv.vocab)
    print("Training set Created")
    train_x, train_y = create_training(tweet_words)
    print("LSTM set")
    mod = setLSTM(vocab_size, weights)
    print("Fitting Model")
    model = fit_model(mod, train_x, train_y)
    # file_name = 'generated_tweets.csv'
    # df.to_csv(file_name)
    return file_name
    

def idx2word(idx):
    return model.wv.index2word[idx]
#gets index when passed word
def word2idx(word):
    if (word in main_control.model.wv.vocab):
        return main_control.model.wv.vocab[word].index
    return 0
#gets word when passed index
def idx2word(idx):
    return main_control.model.wv.index2word[idx]
#creates vectors with 0s of specific sizes
def create_training(tweet_words):
    train_x = np.zeros([len(tweet_words), main_control.max_sent_len],dtype=np.int32)
    train_y = np.zeros([len(tweet_words)],dtype=np.int32)
    #gets indexes
    for i,sentence, in enumerate(tweet_words):
        for t, word in enumerate(sentence[:-1]):
            if word in main_control.model.wv.vocab:
                train_x[i,t] = word2idx(word)
            train_y[i] = word2idx(word)
    return train_x, train_y
#creates the LSTM model. Defines parameters
def setLSTM(vocab_size, weights):
    mod = Sequential()
    mod.add(Embedding(input_dim=vocab_size,output_dim=100,weights=[weights]))
    mod.add(LSTM(units=100))
    mod.add(Dense(units=vocab_size))
    mod.add(Activation('softmax'))
    mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    return mod
#returns probailities after performing math on each item
def sample(preds,temperature=1.0):
    if temperature <=0:
        return np.argmax(preds)
    preds = np.asarray(preds.astype('float64'))
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)
#generates tweets

# maps words to index
def word2idx(word):
    if (word in model.wv.vocab):
        return model.wv.vocab[word].index
    return 0

#runs the model
def fit_model(mod, train_x, train_y):
    mod.fit(train_x, train_y,
            batch_size=200,
            epochs=1)
    model_json = mod.to_json()
    with open ("model.json", "w") as json_file:
        json_file.write(model_json)
    mod.save_weights("model.h5")
    print("model saved to disk")
    
#writes file
def write_file(fin_list):
    df = pd.DataFrame(fin_list, columns=['sample','generated_tweet'])
    return df
main_control("one", "two", "three")