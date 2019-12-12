import os

class Config(object):
    SECRET_KEY = os.environ.get('SECREt_KEY') or 'you-will-never-guess'