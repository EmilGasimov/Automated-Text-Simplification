import sys

def build(filename, train_percent, test_percent):
		with open(filename) as f:
				dataset = f.readlines()
		n = len(dataset)
		train_len = int(n * int(train_percent) / 100)
		test_len = int(n * int(test_percent) / 100)
		with open('train.txt', 'w') as train:
				for train_ex in dataset[:train_len]:
						train.write(train_ex)
		with open('test.txt', 'w') as test:
				test_end = train_len + test_len
				for test_ex in dataset[train_len : test_end]:
						test.write(test_ex)
		with open('valid.txt', 'w') as valid:
				for valid_ex in dataset[test_end:]:
						valid.write(valid_ex)
						
build(sys.argv[1], sys.argv[2], sys.argv[3])
