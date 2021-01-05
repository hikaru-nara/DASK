from utils.utils import extract_word_freq, sentiment_score_init
from transformers import BertTokenizer

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
		source_word_freq = extract_word_freq(source_labeled_text + source_unlabeled_text)
		self.source_freq = source_word_freq.copy()

		target_unlabeled_text = target_data['unlabeled']['text']
		target_word_freq = extract_word_freq(target_unlabeled_text)
		self.target_freq = target_word_freq.copy()

		source_high_freq = [word for word, freq in source_word_freq.items() if freq>=self.min_occurrence]
		target_high_freq = [word for word, freq in target_word_freq.items() if freq>=self.min_occurrence]
		common_words = [word for word in source_high_freq if word in target_high_freq]
		self.common_words = common_words

		# step2: calculate sentiment score of those common words in source domain
		word_count, word_sentiment_count = sentiment_score_init(source_labeled_text,source_label)

		sentiment_score = {}
		for word in common_words:
			if word in word_count.keys():
				sentiment_score[word] = float(word_sentiment_count[word])/float(word_count[word])
			else:
				sentiment_score[word] = 0

		# step3: save to the source and target dicts respectively
		self.source_dict = sentiment_score.copy()
		self.target_dict = {w:s for w,s in sentiment_score.items()  if w in common_words}
		assert 'parking' in common_words
		assert 'parking' in self.source_dict.keys()
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

	def update(self, sentences, pred_labels):
		'''
		update the memory bank per $self.update_steps
		考虑是否加新词；如果加的话还要再maintain一个frequency list
		also update frequent words
		
		'''
		if self.curr_steps == 0:
			# do update
			pass
		else:
			self.curr_steps = (self.curr_steps+1)%self.update_steps
		
		# final step
		self.get_pivots()
