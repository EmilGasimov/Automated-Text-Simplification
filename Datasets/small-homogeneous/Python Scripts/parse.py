import sys

"""
def sentence_reader(filename, i):
    with open(filename) as f:
        return f.readlines()[int(i) - 1].strip()
"""

def parser(norm, simp):
    with open(norm) as f1:
        norm_script = [sent.strip() for sent in f1.readlines()]
    with open(simp) as f2:
        simp_script = [sent.strip() for sent in f2.readlines()]
    return [f'text:{norm_sent}\tlabels:{simp_sent}\tepisode_done:True' for (norm_sent, simp_sent) in zip(norm_script, simp_script)]
        
def write_to_file(filename, dataset):
	with open(filename, 'w') as f:
		  for train_ex in dataset:
		      f.write(train_ex + '\n')
        

dataset = parser(sys.argv[1], sys.argv[2])       
filename = sys.argv[3]
write_to_file(filename, dataset)

