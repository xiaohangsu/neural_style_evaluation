import os
import sys
test_file = '/home/xiaohangsu/caffe/data/flickr_style/test.txt'
test_path = '/home/xiaohangsu/caffe/data/flickr_style/'
print(os.listdir(test_path + sys.argv[1]))

test = open(test_file, 'w')

for filename in os.listdir(test_path + sys.argv[1]):
	test.write(test_path + sys.argv[1] + '/' +filename + ' 1\n')

test.close()