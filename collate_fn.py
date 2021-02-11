import torch
import numpy as np
# import transformers
from uer.utils.constants import *
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def kbert_two_stage_collate_fn(batch_data):
	max_length = batch_data[0]['tokens'].shape[0]
	for i,data in enumerate(batch_data):
		tokens_kg = data.pop('tokens')
		mask_kg = data.pop('mask')
		batch_data[i]['tokens_kg'] = tokens_kg
		batch_data[i]['mask_kg'] = mask_kg

		# tokens_org = tokenizer.encode(data['text'], add_special_tokens=True, max_length=max_length,truncation=True)
		tokens_org = batch_data[i]['tokens_org']
		pad_num = max_length-len(tokens_org)
		if pad_num>0:
			tokens_org.extend([PAD_ID for _ in range(pad_num)])
			if 'ssl_label' in batch_data[i]:
				batch_data[i]['ssl_label'].extend([-1 for _ in range(pad_num)])
		mask_org = [1 if tokenid!=0 else 0 for tokenid in tokens_org]
		batch_data[i]['mask_org'] = np.array(mask_org)
		batch_data[i]['tokens_org'] = np.array(tokens_org)
		batch_data[i]['ssl_label'] = np.array(batch_data[i]['ssl_label'])

	keys = list(batch_data[0].keys())
	batch_data_collated = {}
	for k in keys:
		batch_data_collated[k] = [data[k] for data in batch_data]

	for k in batch_data_collated:
		if isinstance(batch_data_collated[k][0], np.ndarray):
			batch_data_collated[k] = torch.tensor(batch_data_collated[k], dtype=torch.long)
		elif isinstance(batch_data_collated[k][0], int) or isinstance(batch_data_collated[k][0], float)\
			or isinstance(batch_data_collated[k][0], np.int32) or isinstance(batch_data_collated[k][0], np.float32):
			batch_data_collated[k] = torch.tensor(batch_data_collated[k], dtype=torch.long)

	return batch_data_collated

def kbert_two_stage_collate_fn_2item(data_list):
	data_list1 = [datum[0] for datum in data_list]
	data_list2 = [datum[1] for datum in data_list]
	return kbert_two_stage_collate_fn(data_list1), kbert_two_stage_collate_fn(data_list2)

def kbert_ssl_collate_fn_3(data_list):
	data_list1 = [datum[0] for datum in data_list]
	data_list2 = [datum[1] for datum in data_list]
	data_list3 = [datum[2] for datum in data_list]
	return kbert_two_stage_collate_fn(data_list1), kbert_two_stage_collate_fn(data_list2), \
		kbert_two_stage_collate_fn(data_list3)

# def kbert_ssl_collate_fn_3(data_list):



collate_factory_train={
	'sentim': None,
	'causal': None,
	'base_DA': None,
	'base_DA_roberta': None,
	'kbert_two_stage_sentim': kbert_two_stage_collate_fn,
	'kbert_two_stage_da': kbert_two_stage_collate_fn_2item,
	'DANN_kbert': None,
	'DANN_kroberta': None,
	'SSL_kbert': kbert_ssl_collate_fn_3,
	'SSL_kroberta': kbert_ssl_collate_fn_3,
	'SSL_kbert_DANN': kbert_ssl_collate_fn_3,
	'masked_SSL_kbert': None,
	'masked_SSL_kroberta': None
}

collate_factory_eval={
	'sentim': None,
	'causal': None,
	'base_DA': None,
	'base_DA_roberta': None,
	'kbert_two_stage_sentim': kbert_two_stage_collate_fn,
	'kbert_two_stage_da': kbert_two_stage_collate_fn,
	'DANN_kbert': None,
	'DANN_kroberta': None,
	'SSL_kbert': kbert_two_stage_collate_fn,
	'SSL_kroberta': kbert_two_stage_collate_fn,
	'SSL_kbert_DANN': kbert_two_stage_collate_fn,
	'masked_SSL_kbert': kbert_two_stage_collate_fn,
	'masked_SSL_kroberta': kbert_two_stage_collate_fn
}