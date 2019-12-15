import gensim
import numpy as np
import pandas as pd

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.utils.data_utils import get_file

fin_list = []
#gets tweets from file
tweets = [line.rstrip('\n') for line in open('tweets_edited.txt',encoding='utf-8')]
#gets words from tweets for gensim, then removes blank lists
tweet_words = [tweet.lower() for tweet in tweets]
tweet_words = filter(None, tweet_words)
tweet_words = [tweet.split() for tweet in tweet_words]
#creates gensim model
model = gensim.models.Word2Vec(tweet_words,min_count=2)
model.save('model.bin')
model.train(tweet_words,total_examples=model.corpus_count,epochs=10)
weights = model.wv.vectors
#Basic data about dataset
word_count = lambda sentence: len(sentence)
max_sent_len = len(max(tweets,key=word_count))
vocab_size = len(model.wv.vocab)
print('embedding dimensions: ',weights.shape)
print('vocab size: ', len(model.wv.vocab))
#gets index when passed word
def word2idx(word):
    if (word in model.wv.vocab):
        return model.wv.vocab[word].index
    return 0
#gets word when passed index
def idx2word(idx):
    return model.wv.index2word[idx]
#creates vectors with 0s of specific sizes
train_x = np.zeros([len(tweet_words),max_sent_len],dtype=np.int32)
train_y = np.zeros([len(tweet_words)],dtype=np.int32)
#gets indexes
for i,sentence, in enumerate(tweet_words):
    for t, word in enumerate(sentence[:-1]):
        if word in model.wv.vocab:
            train_x[i,t] = word2idx(word)
        train_y[i] = word2idx(word)
print('train_x shape: ', train_x.shape)
print('train_y shape: ', train_y.shape)
#creates the LSTM model. Defines parameters
mod = Sequential()
mod.add(Embedding(input_dim=vocab_size,output_dim=100,weights=[weights]))
mod.add(LSTM(units=100))
mod.add(Dense(units=vocab_size))
mod.add(Activation('softmax'))
mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
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
def generate_next(text,num_generated = 10):
    word_idxs = [word2idx(word) for word in text.lower().split()]
    for i in range(num_generated):
        prediction = mod.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature = 0.7)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx) for idx in word_idxs)
#when each epoch is over use these phrases to make predictions
def on_epoch_end(epoch,_):
    texts = [
            'big game player',
            'swing',
            'the key to life is',
            'movitation',
            ]
    for text in texts:
        sample = generate_next(text)
        print('%s ..-> %s' %(text,sample))
        temp = [text,sample]
        fin_list.append(temp)

#runs the model
mod.fit(train_x, train_y,
        batch_size=100,
        epochs=1,
        callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
#writes file
df = pd.DataFrame(fin_list, columns=['sample','generated_tweet'])
df.to_csv('generated_tweets_test.csv')
