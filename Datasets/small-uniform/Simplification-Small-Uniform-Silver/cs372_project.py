import time
before_import = time.time()
import nltk
from nltk.corpus import brown, gutenberg, conll2000
from nltk.corpus import wordnet as wn
import pickle
import sys

after_import = time.time()

with open("dictionary.pickle", "rb") as f:
		dictionary = pickle.load(f)
		
train_data = conll2000.tagged_sents('train.txt') + brown.tagged_sents(categories=['news'])
test_data = conll2000.tagged_sents('test.txt')

patterns = [
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # simple past
    (r'.*es$', 'VBZ'),                 # 3rd singular present
    (r'.*ould$', 'MD'),                # modals
    (r'.*\'s$', 'NN$'),                # possessive nouns
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                      # nouns (default)
]

regexp_tagger = nltk.RegexpTagger(patterns)
unigram_tagger = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)

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

all_words = words("unigram_freq.csv")

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos, 
            "prevpos+pos": "%s+%s" % (prevpos, pos),  
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}  

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))



class ConsecutiveNPChunkTagger(nltk.TaggerI): 

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm=None, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt')
train_sents = conll2000.chunked_sents('train.txt')
chunker = ConsecutiveNPChunker(train_sents)


from urllib.request import urlopen
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.wsd import lesk
import json
import re
import inflect

engine = inflect.engine()

def simplify(sent, bigrams):
    sents = []
    start = 0
    try:
        for i in range(len(bigrams)):
            pair = bigrams[i]
            w1, t1 = pair[0]
            w2, t2 = pair[1]
            if w1 == ',' and t2.startswith('W'):
                sents.append(sent[start:i])
                start = i + 1
                end = start
                while end < len(sent) and sent[end] != ',':
                    end += 1
                sents.append(sent[start:end])
                if end < len(sent):
                    sents[-2].extend(sent[end + 1:])
                sents[-1][0] = 'He/she/it/they/there'
                break
        if not sents:
            sents.append(sent)
    except:
        sents.append(sent)
    return sents
    
def refine(sent):
		try:
				if sent[-1] == ',':
				    sent[-1] = '.'
				if not sent[0].istitle():
				    sent[0] = sent[0].title()
				if sent[-1].isalpha():
				    sent.append('.')
				return sent
		except:
				return sent

def decompose(sent, tagger):
    sents = []
    tagged_sent = tagger.tag(sent)
    bigrams = list(nltk.trigrams(tagged_sent))
    start = 0
    for i in range(start, len(bigrams)):
        triple = bigrams[i]
        w1, t1 = triple[0]
        w2, t2 = triple[1]
        w3, t3 = triple[2]
        cond = ((t2 == 'RB' or t2 == 'PRP' or t2.startswith('NN')) and (t3.startswith('VB') or t3 == 'BEDZ'))
        if t1 == 'CC' and cond:
            sents.append(sent[start:i])
            sent = sent[i + 1:]
            start = i + 1
        elif i == len(bigrams) - 1:
            sents.append(sent)
    output = []
    for sent in sents:
        output.append(refine(sent))
        
    simplified_sents = []
    length_so_far = 0
    for sent in sents:
        for simple_sent in simplify(sent, bigrams[length_so_far : length_so_far + len(sent)]):
            simplified_sents.append(simple_sent)
        length_so_far += len(sent)
    
    simplified_sents = [refine(sent) for sent in simplified_sents]
    return simplified_sents

conversion_map = defaultdict(list)
conversion_map['adjective'] = ['a', 's']
conversion_map['adverb'] = ['r']
conversion_map['noun'] = ['n']
conversion_map['verb'] = ['v']
conversion_map['plural noun'] = ['n']
conversion_map['transitive verb'] = ['v']
stopwords = stopwords.words('english')


def freq_index(word, all_words):
    for i in range(len(all_words)):
        if all_words[i] == word:
            return i
    return -1

# word sense disambiguation
def wsd(sent, word):
    return lesk(sent, word)

"""
def define(word):
    url = 'https://api.dictionaryapi.dev/api/v2/entries/en_US/' + word
    try:
        data = urlopen(url).read()
        dictionary = json.loads(data)
        return dictionary[0]
    except:
        return None
"""

def define(word):
	if word in dictionary:
		return dictionary[word]
	else:
		return None		

def is_list_of_dicts(d):
    return isinstance(d, list) and all(isinstance(item, dict) for item in d)


def get_synonyms(word):
    entry = define(word)
    if not entry:
        return None
    meanings = entry['meanings']
    output = defaultdict(list)
    for meaning in meanings:
        pos = meaning['partOfSpeech']
        for definition in meaning['definitions']:
            if 'synonyms' in definition:
                output[pos].append([definition['definition'], definition['synonyms']])
    return output
		

def same_lemma(word1, word2):
    try:
        def1, def2 = define(word1), define(word2)
        return def1['word'] == def2['word']
    except:
        return False

def similarity_score(word_def, syn_def, syn_list):
    for syn in syn_list:
        index = freq_index(syn, all_words)
        if (syn in word_def) and (index > -1) and (index < 10000):
            return syn
    sent1 = nltk.word_tokenize(word_def)
    sent2 = nltk.word_tokenize(syn_def)
    tagged_sent1 = nltk.pos_tag(sent1)
    tagged_sent2 = nltk.pos_tag(sent2)
    word_def = [(word.lower(), tag) for (word, tag) in tagged_sent1 if word.isalpha() and not word in stopwords]
    syn_def = [(word.lower(), tag) for (word, tag) in tagged_sent2 if word.isalpha() and not word in stopwords]
    score = 0
    for (word1, tag1) in word_def:
        for (word2, tag2) in syn_def:
            if same_lemma(word1, word2):
                score += 1
                if tag1.startswith('V') and tag2.startswith('V'):
                    score += 0.25
    return score


f = open('irregular_verbs.txt')
irreg_verbs = {}
for line in f:
    forms = line.split()
    verb = forms[0]
    vbd = forms[1]
    vbn = forms[2]
    vbg = (forms[0] + 'ing') if not (forms[0][-1] in 'aeiou') else (forms[0][:-1] + 'ing')
    vbz = forms[0] + 's' #(forms[0] + 'es') if not (forms[0][-1] in 'aeiou') else (forms[0] + 's')
    irreg_verbs[verb] = {'VBD': vbd, 'VBZ': vbz, 'VBG': vbg, 'VBN': vbn, 'VBP': verb}
    
def pluralize(noun):
    return engine.plural(noun)


def comparative(adj):
    if len(adj) > 5:
        return adj
    return (adj + 'r') if adj[-1] == 'e' else (adj + 'er')
    
    
def superlative(adj):
    if len(adj) > 5:
        return adj
    return (adj + 'st') if adj[-1] == 'e' else (adj + 'est')
    
            
def handle_word(original_word, original_tag, letter, sub_word):
    try:
        cond1 = original_tag.startswith('JJ') and letter == 's'
        cond2 = original_tag.lower().startswith(letter)
        if cond2:
            if letter == 'v':
                for end in ['ed', 'ing']:
                    if original_word.endswith(end):
                        returnVal = (sub_word + end if sub_word[-1]
                                != 'e' else sub_word[:-1] + end)
                    return returnVal
                return irreg_verbs[sub_word][original_tag]
            elif letter == 'n':
                return sub_word
        elif cond1:
            if original_word.endswith('er'):
                return comparative(sub_word)
            elif original_word.endswith('est'):
                return superlative(sub_word)
            else:
                return sub_word
        return sub_word
    except:
        return sub_word


def easiest_synonym(tagged_sent, i, all_words):
    word = tagged_sent[i][0]
    basic_form = word
    d = define(word)
    if d:
        basic_form = d['word']
    fi = freq_index(basic_form, all_words)
    if fi < 10000:
        return word
    elif word.isalpha() and i > 0 and i < len(tagged_sent) and word.istitle():
        return word
    
    synset = wsd([w for (w, t) in tagged_sent], word)
    if synset:
        word_tag = synset.pos()
        word_definition = synset.definition()
        synonyms = get_synonyms(word) # returns a map, where keys are pos-tags
        if synonyms:
            easiest = None
            index = 333333
            for pos_tag in synonyms:
                if word_tag in conversion_map[pos_tag] and (tagged_sent[i][1].lower().startswith(word_tag) or 
                                                            (tagged_sent[i][1].startswith('JJ') and word_tag == 's')):
                    max_score = 0
                    syn_list = []
                    for n in range(len(synonyms[pos_tag])):
                        definition, synonym_list = synonyms[pos_tag][n]
                        score = similarity_score(word_definition, definition, synonym_list)
                        if isinstance(score, str):
                            return score
                        elif score > max_score:
                            max_score = score
                            syn_list = synonym_list
                    if max_score == 0 and len(synonyms[pos_tag]) == 1:
                        for j in range(len(synonyms[pos_tag][0][1])):
                            current_index = freq_index(synonyms[pos_tag][0][1][j], all_words)
                            if current_index > -1 and current_index < 10000:
                                easiest = synonyms[pos_tag][0][1][j]  
                    else:   
                        for synonym in syn_list:
                            current = freq_index(synonym, all_words)
                            if current > -1 and current < 10000:
                                easiest = synonym
                                break
                    easiest = handle_word(word, tagged_sent[i][1], word_tag, easiest)
            if easiest:
                return easiest
    return word

def simplify_text(text, tagger, all_words):
    sents = nltk.sent_tokenize(text)
    tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]
    modified_sents = []
    for sent in tokenized_sents:
        a = decompose(sent, tagger)
        for simple_sent in a:
            modified_sents.append(simple_sent)
    output = [[easiest_synonym(tagger.tag(sent), i, all_words) for i in range(len(sent))] 
            for sent in modified_sents]
    return output

def simplify_and_convert_to_text(text, tagger, all_words):
    sents = simplify_text(text, tagger, all_words)
    text = ''
    for sent in sents:
        for i in range(len(sent)):
            if sent[i] == "'s":
                text = text[:-1]
            text += sent[i]
            if (i < len(sent) - 1 and not (sent[i + 1] in '.,?!:;')) or i == len(sent) - 1:
                text += ' '
    return text

class NLTK_Tagger(nltk.TaggerI): 
    def tag(self, sentence):
        return nltk.pos_tag(sentence)
    
nltk_tagger = NLTK_Tagger()


print(f'Import statements took {after_import - before_import} seconds.')

start = time.time()
print(f'Other than import statements but before calling the main function: {start - after_import} seconds')


input_file, output_file = sys.argv[1], sys.argv[2]

with open(input_file) as f:
    examples = [example.split('\t') for example in f.readlines()]
    n = len(examples)

i = 1
with open(output_file, 'w') as output: 
    for example in examples:
        orig_sent = example[0][5:]
        episode_done = example[2]
        print('\n')
        print(f'SENTENCE NUMBER: {i}/{n}')
        print('ORIGINAL SENTENCE:', orig_sent)
        start = time.time()
        simplified_sentence = simplify_and_convert_to_text(orig_sent, nltk_tagger, all_words)
        end = time.time()
        print('SIMPLIFIED SENTENCE:', simplified_sentence)
        print(f'It took {end - start} seconds.')
        output.write(f'text:{orig_sent}\tlabels:{simplified_sentence}\t{episode_done}')
        print('=======================================')
        i += 1





