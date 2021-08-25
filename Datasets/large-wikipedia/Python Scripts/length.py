import sys

def length(filename):
	with open(filename) as f:
		return len(f.readlines())
		
print(length(sys.argv[1]))
