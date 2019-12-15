from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
#import io

ckey = 'SVW2t6gxefmiNqWMFx21vcyAJ'
csecret = 'pzvHtfQMoq3R69Lyh1wWCx2Zuq3nsJj7HO4ZD48yJI7lLD8Jip'
atoken = '1113510609981792261-SBsQWa22LcXoplME78Fnh5C7c2qxIf'
asecret = 'aoo5z6bLQqs8PcevqOyqbU5RkXOyhGULGcrVFaG5xEVqk'

class listener(StreamListener):
	def on_data(self,data):
		#print(data)
		data = json.loads(data)
		language = data['lang']
		text = data['text']
		if language == 'en':			
			file.write(text)
			#differentiates between tweets
			file.write('\n')
		return True
	def on_error(self,status):
		if status == 420:
			return False
		print(status)
'''
def file_len(fname):
	with open(fname) as f:
		for i,l in enumerate(f):
			pass
	return i + 1
'''
auth  = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken,asecret)
#tweets = []
with open('tweets.txt','a+',encoding='utf-8') as file:
#categories = ['Game of Thrones','django','python','airplane']
	twitterStream = Stream(auth,listener())
	twitterStream.filter(track=['football', 'soccer', 'volleyball'])
