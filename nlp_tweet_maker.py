# import nlp_model_save
from keras.models import model_from_json
from keras import backend as K
import gensim
import numpy as np
import random

# maps words to index
def word2idx(word):
    if (word in use_saved_model.model.wv.vocab):
        return use_saved_model.model.wv.vocab[word].index
    return 0
def idx2word(idx):
    return use_saved_model.model.wv.index2word[idx]

def random_num():
    rand = random.randint(2, 14)
    return rand

def use_saved_model():
    # Loads Keras model
    json_file = open('model.json')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Loads Weights
    loaded_model.load_weights("model.h5")
    # Loads Gensim model
    model = gensim.models.Word2Vec.load('model.bin')
    # model.train(tweet_words,total_examples=model.corpus_count,epochs=10)
    use_saved_model.model = model
    return loaded_model
def control(texts):
    K.clear_session()
    loaded_model = use_saved_model()
    
    # <input id="referencephrase1" name="referencephrase1" required type="text" value="Chelsea"> 
    # print("texts: ", texts)
    generated_tweets = []
    for text in texts:
        str(text)
        sample = generate_next(text, loaded_model)
        # print('%s ..-> %s' %(text,sample))
        generated_tweets.append(sample)
    # print(generated_tweets)
    K.clear_session()
    return generated_tweets


def generate_next(text, mod,num_generated = 10):
    word_idxs = [word2idx(word) for word in text.lower().split()]
    num_generated = random_num()
    for i in range(num_generated):
        prediction = mod.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature = 0.7)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx) for idx in word_idxs)

def sample(preds,temperature=1.0):
    if temperature <=0:
        return np.argmax(preds)
    preds = np.asarray(preds.astype('float64'))
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)