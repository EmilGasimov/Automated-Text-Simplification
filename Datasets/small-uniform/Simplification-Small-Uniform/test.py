import pickle
import sys

with open('dictionary.pickle', 'rb') as f:
		d = pickle.load(f)
		
word = sys.argv[1]
if word in d:
		print(d[word]['meanings'][0])
else:
		print('not in dictionary')
