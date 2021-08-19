import sys
import random

def parse(filename):
  with open(filename) as f:
    return [line.strip() for line in f.readlines()]

def build_dataset(norm, simps):
  norm = parse(norm)
  simps = [parse(simp) for simp in simps]
  simps = list(zip(*simps))
  return list(zip(norm, simps))

def write_to_file(filename, dataset):
  with open(filename, 'w') as f:
    for (norm, simps) in dataset:
      for i in range(len(simps)):
        if i < len(simps) - 1:
          f.write(f'text:{norm}\tlabels:{simps[i]}\tepisode_done:False\n')
        else:
          f.write(f'text:{norm}\tlabels:{simps[i]}\tepisode_done:True\n')

test_dataset = build_dataset(sys.argv[1], sys.argv[2:11])
tune_dataset = build_dataset(sys.argv[11], sys.argv[12:21])
filename = sys.argv[21]
dataset = test_dataset + tune_dataset
random.shuffle(dataset)
write_to_file(filename, dataset)
