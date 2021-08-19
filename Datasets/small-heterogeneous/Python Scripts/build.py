import sys

def build(filename, train_percent, test_percent, num_of_simp_versions=9):
		with open(filename) as f:
				dataset = f.readlines()
		n = len(dataset)
		train_len = int(n * int(train_percent) / 100)
		train_len = train_len - (train_len % num_of_simp_versions)
		test_len = int(n * int(test_percent) / 100)
		test_len = test_len - (test_len % num_of_simp_versions)
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
