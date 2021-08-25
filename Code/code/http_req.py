import asyncio
import aiohttp
from aiohttp import ClientSession, ClientConnectorError
import json
import pickle
import time
from nltk.corpus import wordnet as wn


# storing the most frequent n words, in the decreasing order of frequency
def words(filename):
    f = open(filename)
    output = []
    i = 0
    for line in f:
        if i == 0:
            i = 1
            continue
        word, freq = line.split(",")
        output.append(word)
        i = 1
    f.close()
    return output

all_words = words("unigram_freq.csv")[:100000]
d = {}

async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        resp = await session.request(method="GET", url=url, **kwargs)
        resp_body = await resp.read()
        warning = 'WARNING' if (int(resp.status) == 429) else ''
        if warning:
        	all_words.append(url.split('/')[-1])
        print(resp.status, warning)
        return json.loads(resp_body)[0]
    except:
        return None


async def make_requests(urls: set, **kwargs) -> None:
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_html(url=url, session=session, **kwargs))
        results = await asyncio.gather(*tasks)

    for result in results:
    		if result:
        		d[result["word"]] = result
        

if __name__ == "__main__":
	import pathlib
	import sys

	assert sys.version_info >= (3, 7), "Script requires Python 3.7 ."
	here = pathlib.Path(__file__).parent
	url = "https://api.dictionaryapi.dev/api/v2/entries/en_US/"
	i = 0
	n = len(all_words) 
	batch_size = 100
	#short_interval, long_interval = 4, 40
	interval = 30
	start = None
	k = 0
	while k < int(len(all_words) / batch_size) + 1:
		end = time.time()
		if start:
			print(f'{end - start} seconds passed.')
			print(f'batch {k} completed.')

		print('============================================\n')
		print(f'batch {k + 1} started.')
		wordlist = all_words[i : i + batch_size]
		i += batch_size
		urls = set(url + word for word in wordlist)
		asyncio.run(make_requests(urls=urls))
		with open("dictionary.pickle", "wb") as f:
			pickle.dump(d, f)
		start = time.time()
		#interval = long_interval if (k % 5 == 4) else short_interval
		time.sleep(interval)
		print(f'remaining words: {len(all_words) - (k + 1) * batch_size}')
		print(f'in percent : {(len(all_words) - (k + 1) * batch_size) / n * 100}%')
		print(f'dictionary length: {len(d)}')
		k += 1
		


