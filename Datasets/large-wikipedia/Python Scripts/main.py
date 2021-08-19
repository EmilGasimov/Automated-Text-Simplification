import sys
import random

def parse(filename):
  with open(filename) as f:
    triples = [line.split('\t') for line in f.readlines()]
    return [sent.strip() for (title, par_num, sent) in triples]

def build(filename, norm, simp):
  norm_sents = parse(norm)
  simp_sents = parse(simp)
  aligned_sents = list(zip(norm_sents, simp_sents))
  random.shuffle(aligned_sents)
  with open(filename, 'w') as f:
    for (norm_sent, simp_sent) in aligned_sents:
      f.write(f'text:{norm_sent}\tlabels:{simp_sent}\tepisode_done:True\n')

filename, norm, simp = sys.argv[1], sys.argv[2], sys.argv[3]
build(filename, norm, simp)

