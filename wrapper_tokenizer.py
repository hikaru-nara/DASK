from fairseq.data.encoders.gpt2_bpe_utils import Encoder
from fairseq.file_utils import cached_path
import json

DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

class gpt2_tokenizer(object):
	'''
	reference https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/gpt2_bpe.py
	'''
	def __init__(self, vocab=DEFAULT_VOCAB_BPE, encoder_json_path=DEFAULT_ENCODER_JSON):
		'''
		vocab is a dictionary {word:number} or a path to the vocabulary file
		'''
		# vocab is path to the vocabulary file
		if isinstance(vocab,str):
			vocab_bpe_path = cached_path(vocab)
			with open(vocab_bpe_path, 'r', encoding="utf-8") as f:
				bpe_data = f.read()
				print('wrapper_tokenizer vocab')
				print(bpe_data.split('\n')[1].encode('utf-8'))
				# bpe_merges is list of (word, number)
				bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
		# vocab is dictionary
		else:
			bpe_data = vocab
			bpe_merges = bpe_data.items()

		encoder_json = cached_path(encoder_json_path)
		with open(encoder_json, 'r') as f:
			encoder = json.load(f)

		assert isinstance(encoder, dict)
		self.inverse_vocab = {bpe[1]:bpe[0] for bpe in bpe_merges}
		self.bpe = Encoder(encoder, bpe_merges)

	def encode(self, x: str) -> str:
		'''
		return encoded token_ids converted into strings joined by ' '
		sample output: '111 222 333'
		'''
		return ' '.join(map(str, self.bpe.encode(x)))

	def decode(self, x: str) -> str:
		return self.bpe.decode([
			int(tok) if tok not in {'<unk>', '<mask>'} else tok
			for tok in x.split()
		])

	def cut(self, x:str):
		'''
		return list of raw-word tokens
		without filtering the \t and \n 
		'''
		def convert(string):
			return bytearray([self.bpe.byte_decoder[c] for c in string]).decode('utf-8', errors=self.bpe.errors)
		# print('wt')
		# print(x)
		text = [self.bpe.decoder.get(token, token) for token in self.bpe.encode(x)]
		text = [convert(token) for token in text]
		# print('wt')
		# print(text)
		return text

	def is_beginning_of_word(self, x: str) -> bool:
		return self.decode(x).startswith(' ')


		
