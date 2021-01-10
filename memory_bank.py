from utils.utils import extract_word_freq, sentiment_score_init
from transformers import BertTokenizer
import os
import pickle as pkl
class MemoryBank(object):
	def __init__(self, args):
		self.args = args
		self.source = args.source
		self.target = args.target
		self.update_steps = args.update_steps
		self.min_occurrence = args.min_occur
		self.num_pivots = args.num_pivots
		self.source_dict = {} # maintain the sentiment score of each word, range [-1,1]
		self.target_dict = {}
		self.source_freq = {}
		self.target_freq = {}
		self.pivot2token = {}
		self.common_words = []
		self.curr_steps = 0
		self.update_freq = True
		self.pivots = []
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.alpha = args.update_rate
		self.conf_threshold = args.confidence_threshold
		# subword pivot?

	def initialize(self, source_data, target_data):
		'''
		initialize the memory bank before training
		source_data/target_data: return value from readers
			format: {"labeled": labeled_data, "unlabeled": unlabeled_data} 
		'''
		# step1: extract common words in both domains
		source_labeled_text = source_data['labeled']['text']
		source_label = source_data['labeled']['label']
		source_unlabeled_text = source_data['unlabeled']['text']
		source_freq_filename = 'data/amazon-review-old/{}/all_freq.pkl'.format(self.source.split('.')[-1]) # filename needs generalized
		if os.path.exists(source_freq_filename):
			with open(source_freq_filename, 'rb') as f:
				source_word_freq = pkl.load(f)
		else:
			source_word_freq = extract_word_freq(source_labeled_text + source_unlabeled_text)
			with open(source_freq_filename, 'wb') as f:
				pkl.dump(source_word_freq, f)
		self.source_freq = source_word_freq.copy()

		target_freq_filename = 'data/amazon-review-old/{}/un_freq.pkl'.format(self.target.split('.')[-1])
		target_unlabeled_text = target_data['unlabeled']['text']
		if os.path.exists(target_freq_filename):
			with open(target_freq_filename, 'rb') as f:
				target_word_freq = pkl.load(f)
		else:
			target_word_freq = extract_word_freq(target_unlabeled_text)
			with open(target_freq_filename, 'wb') as f:
				pkl.dump(target_word_freq, f)
		self.target_freq = target_word_freq.copy()

		source_high_freq = [word for word, freq in source_word_freq.items() if freq>=self.min_occurrence]
		target_high_freq = [word for word, freq in target_word_freq.items() if freq>=self.min_occurrence]
		# common_words = [word for word in source_high_freq if word in target_high_freq]
		common_words = list(set(source_high_freq) & set(target_high_freq))
		self.common_words = common_words

		# step2: calculate sentiment score of those common words in source domain
		sentiment_score_filename = 'data/amazon-review-old/{}/sentiment.pkl'.format(self.source.split('.')[-1])
		if os.path.exists(sentiment_score_filename):
			with open(sentiment_score_filename, 'rb') as f:
				sentiment_score = pkl.load(f)
		else:
			word_count, word_sentiment_count = sentiment_score_init(source_labeled_text,source_label)
			# print(word_sentiment_count['+'], word_count['+'])
			# print('commented')
			sentiment_score = {}
			for word in common_words:
				if word in word_count.keys() and word_count[word]>self.min_occurrence:
					sentiment_score[word] = float(word_sentiment_count[word])/float(word_count[word])
				else:
					sentiment_score[word] = 0
			with open(sentiment_score_filename, 'wb') as f:
				pkl.dump(sentiment_score, f)

		# step3: save to the source and target dicts respectively
		self.source_dict = sentiment_score.copy()
		self.target_dict = {w:s for w,s in sentiment_score.items()  if w in common_words}
		self.get_pivots()

	def get_pivots(self):
		'''
		get the pivot words from the two score dicts
		'''
		# step0: collect common words
		# source_high_freq = [word for word, freq in self.source_freq.items() if freq>=self.min_occurrence]
		# target_high_freq = [word for word, freq in self.target_freq.items() if freq>=self.min_occurrence]
		# common_words = [word for word in source_high_freq if word in target_high_freq]
		# self.common_words = common_words

		# step1: calculate the mean score in two domains
		score_dict = {}
		for w in self.common_words:
			score_dict[w] = (self.source_dict[w] + self.target_dict[w])/2

		# step2: return the top $self.num_pivots words

		sorted_word_score = [k for k, v in sorted(score_dict.items(), key=lambda item: -abs(item[1]))]
		self.pivots = sorted_word_score[:self.num_pivots]
		for p in self.pivots:
			if not p in self.pivot2token:
				self.pivot2token[p] = self.tokenizer.encode(p, add_special_tokens=False)
		# assert 'recommendations' in self.pivot2token

	def update(self, sentences, pred_labels, pred_confidence, source_or_target='source', step=False):
		'''
		update the memory bank per $self.update_steps
		考虑是否加新词；如果加的话还要再maintain一个frequency list
		also update frequent words
		TODO: high confidence
		'''
		if self.curr_steps == 0:
			# do update
			if step:
				self.curr_steps += 1
			if source_or_target == 'source':
				sentiment_score = self.source_dict
			else:
				sentiment_score = self.target_dict
			for sentence, label, conf in zip(sentences, pred_labels, pred_confidence):
				unique = set()
				for word in word_tokenize(sentence):
					word = word.lower()
					if word not in unique:
						if conf>=self.conf_threshold:
							# if label == 1:
							sentiment_score[word] = (1-self.alpha)*sentiment_score[word] + self.alpha*(2*label-1)
						else:
							sentiment_score[word] = (1-self.alpha)*sentiment_score[word]

						unique.add(word)
		else:
			if step:
				self.curr_steps = (self.curr_steps+1)%self.update_steps

		
		# final step
		self.get_pivots()

# if __name__ == '__main__':
# 	from utils.readers import reader_factory