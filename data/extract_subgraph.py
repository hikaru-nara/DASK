
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(dir_path))
import wrapper_tokenizer as wt 
import json
from functools import reduce
import requests
import xml.etree.ElementTree as ET
import csv
# from unidecode import unidecode


# domains = ['books', 'dvd', 'electronics', 'kitchen']
# file_list = reduce(lambda x,y:x+y, \
# 				[[os.path.join('amazon-review-old', d, 'positive.parsed'), \
# 				os.path.join('amazon-review-old', d, 'negative.parsed'), \
# 				os.path.join('amazon-review-old', d, '{}UN.txt'.format(d))] for d in domains])

# file_list = [os.path.join('imdb',n) for n in ['train.tsv','dev.tsv']]
# unique_token_list = set()
# tokenizer = wt.gpt2_tokenizer()
# for file in file_list:
# 	with open(file, "r") as f:
# 		reader = csv.reader(f, delimiter="\t", quotechar=None)
# 		for i,line in enumerate(reader):
# 			if i == 0:
# 				# the header of the CSV files
# 				# n += 1
# 				continue

# 			t = line[0]
# 			y = line[1]
# 			# print('imdb/get_examples/label',y)
# 			# print('imdb/get_examples/text',t)
# 			tokens  = tokenizer.cut(t)
# 			# print(tokens)
# 			unique_token_list.update(tokens)
	# tree = ET.parse(file)
	# root = tree.getroot()
	# reviews = root.iter('review')
	# print(file)
	# for review in reviews:
	# 	tokens = tokenizer.cut(review.text)
	# 	# for token in tokens:
	# 	# 	print(token)
	# 	unique_token_list.update(tokens)
# print('token_list length')
# print(len(list(unique_token_list)))
# with open('unique_token_list_imdb.txt', 'w') as f:
# 	for token in list(unique_token_list):
# 		print('WRITE token {}'.format(token))
# 		f.write(token)
# 		f.write('\n')
# 	f.close()


def standardize(word):
	# print(word)
	# print(tmp)
	tmp = word.strip().strip('\'').lower()
	# print(tmp)
	if tmp.find('\'') != -1:
		result = tmp[:tmp.find('\'')]
	else:
		result = tmp
	# print(result)
	return result

# standardize('unfoldings') 


with open('unique_token_list_imdb.txt', 'r') as f:
	text = f.read()
	unique_token_list = [standardize(t) for t in text.split('\n')[:-1]] 
	f.close()
print(len(list(unique_token_list)))

unique_token_list = list(unique_token_list)
print(unique_token_list[:30])
unique_token_list = unique_token_list[5000:10000]
triplet_list = []
for i,token in enumerate(list(unique_token_list)):
	if i%1000==0:
		print('----------{}----------'.format(i))
		# print(i)
	obj = requests.get('http://api.conceptnet.io/c/en/{}'.format(token))
	
	try:
		obj_dict = obj.json()
	except:
		print('except')
		print(token)
		continue
	try:
		edges = obj_dict['edges']
	except:
		print('except edges')
		print(token)
		continue
	for edge in edges:
		subj = edge['start']['@id'].split('/')[3]
		pred = edge['rel']['@id'].split('/')[2]
		obje = edge['end']['@id'].split('/')[3]
		# print(edge['end']['@id'].split('/')[2], edge['start']['@id'].split('/')[2])
		# only english nodes are preserved
		if edge['end']['@id'].split('/')[2]=='en' and edge['start']['@id'].split('/')[2]=='en':
			triplet_list.append((subj,pred,obje))
		# subj = subj[:subj.rfind('/')]
		# subj = subj[subj.rfind('/')+1:]
		# pred = pred[pred.rfind('/')+1:]
		# obje = obje[pred.rfind('/')+1:]

print(triplet_list[:10])
with open('imdb_sub_conceptnet.spo', 'a') as f:
	for triplet in triplet_list:
		f.write(triplet[0]+'\t'+triplet[1]+'\t'+triplet[2]+'\n')
	f.close()