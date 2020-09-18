
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(dir_path))
import wrapper_tokenizer as wt 
import json
from functools import reduce
import requests
import xml.etree.ElementTree as ET


domains = ['books', 'dvd', 'electronics', 'kitchen']
file_lists = reduce(lambda x,y:x+y, \
				[[os.path.join('amazon-review-old', d, 'positive.parsed'), \
				os.path.join('amazon-review-old', d, 'negative.parsed'), \
				os.path.join('amazon-review-old', d, '{}UN.txt'.format(d))] for d in domains])
# unique_token_list = set()
# tokenizer = wt.gpt2_tokenizer()
# for file in file_lists:
# 	tree = ET.parse(file)
# 	root = tree.getroot()
# 	reviews = root.iter('review')
# 	print(file)
# 	for review in reviews:
# 		tokens = tokenizer.cut(review.text)
# 		# for token in tokens:
# 		# 	print(token)
# 		unique_token_list.update(tokens)
# print('token_list length')
# print(len(list(unique_token_list)))
# with open('unique_token_list_2.txt', 'w') as f:
# 	for token in list(unique_token_list):
# 		print('WRITE token {}'.format(token))
# 		f.write(token)
# 		f.write('\n')
# 	f.close()

with open('unique_token_list_2.txt', 'r') as f:
	text = f.read()
	unique_token_list = [t.strip() for t in text.split('\n')[:-1]] 
	f.close()
# unique_token_list = ['parse']
print(len(list(unique_token_list)))
unique_token_list = unique_token_list[45000:]
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
with open('bdek_sub_conceptnet.spo', 'a') as f:
	for triplet in triplet_list:
		f.write(triplet[0]+'\t'+triplet[1]+'\t'+triplet[2]+'\n')
	f.close()