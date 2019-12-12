from app import app
from app.forms import GetTweetForm, GenerateTweetForm
from flask import render_template, flash, redirect
from keras import backend as K

import nlp_tweet_maker

@app.route('/')
@app.route('/index')
def index():
    user = {'username':'Julian'}
    posts = [
        {
            'author':{'username':'Julian'},
            'body': 'Nice day in Ibiza'
        },
        {
            'author':{'username':'Jordan'},
            'body': 'daadadadadaadaadaadaadaa'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts = posts)

# @app.route('/tweetget', methods=['GET', 'POST'])
# def login():
#     form = GetTweetForm()
#     if form.validate_on_submit():
#         flash('Tweets Requested for term {}, for {} time'.format(
#             form.searchterm.data, form.time.data))
#         val = nlp_tweet_collect.getTweets(form.file_name.data, form.time.data, form.searchterm.data)
#         print(val)
#         return redirect('/index')
#     return render_template('tweetform.html', title='Get Tweets', form=form)

@app.route('/tweetgenerate', methods=['GET','POST'])
def generateTweet():
    form = GenerateTweetForm()
    if form.validate_on_submit():
        phrase1 = form.referencephrase1.data
        phrase2 = form.referencephrase2.data
        phrase3 = form.referencephrase3.data
        #print(phrase1, phrase2, phrase3)
        # tweets is a list
        K.clear_session()
        texts = [phrase1, phrase2, phrase3]
        # texts=clean_phrases(texts)
        tweets = nlp_tweet_maker.control(texts)
        return render_template('viewtweet.html', tweets=tweets)
    return render_template('generateform.html', title='Generate Tweets', form = form)