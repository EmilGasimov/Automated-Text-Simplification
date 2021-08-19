import sys
import random

"""
def merge(*fnames):
		merged = [];
		i = 0
		for fname in fnames:
				with open(fname) as f:
						merged.extend(f.readlines())
		return merged
"""

def merge(fname1, fname2):
		with open(fname1) as f1:
				file1 = f1.readlines()
		with open(fname2) as f2:
				file2 = f2.readlines()
		return file1 + file2
		
merged = merge(sys.argv[1], sys.argv[2])
random.shuffle(merged)
filename = sys.argv[3]
with open(filename, 'w') as f:
		for train_ex in merged:
				f.write(train_ex)
