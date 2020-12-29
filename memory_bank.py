from utils.utils import extract_word_freq

class MemoryBank(object):
	def __init__(self, args):
		self.args = args
		self.source = args.source
		self.target = args.target
		self.update_steps = args.update_steps
		self.min_occurrence = args.min_occurrence
		self.num_pivots = args.num_pivots
		self.source_dict = {} # maintain the sentiment score of each word, range [-1,1]
		self.target_dict = {}
		self.curr_steps = 0
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

		target_unlabeled_text = target_data['unlabeled']['text']
		target_word_freq = extract_word_freq(target_unlabeled_text)

		source_high_freq = [word if freq>=self.min_occurrence for word, freq in source_word_freq.items()]
		target_high_freq = [word if freq>=self.min_occurrence for word, freq in target_word_freq.items()]
		common_words = [word if word in target_high_freq for word in source_high_freq]
		# step2: calculate p(y|w) of those common words in source domain
		for sentence in source_labeled_text:
			for word in sentence:
		# step3: save to the source and target dicts respectively

	def get_pivots(self):
		'''
		get the pivot words from the two score dicts
		'''
		# step1: calculate the mean score in two domains

		# step2: return the top $self.num_pivots words

	def update(self, sentences, pred_labels):
		'''
		update the memory bank per $self.update_steps
		'''
		if self.curr_steps == 0:
			# do update

			self.curr_steps = 0
		else:
			self.curr_steps = (self.curr_steps+1)%self.update_steps
		
