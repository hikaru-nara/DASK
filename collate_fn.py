import torch
import numpy as np
# import transformers

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def kbert_two_stage_collate_fn(batch_data):
	max_length = batch_data[0]['tokens'].shape[0]
	for i,data in enumerate(batch_data):
		tokens_kg = data.pop('tokens')
		mask_kg = data.pop('mask')
		batch_data[i]['tokens_kg'] = tokens_kg
		batch_data[i]['mask_kg'] = mask_kg

		tokens_org = tokenizer.encode(data['text'], add_special_tokens=True, max_length=max_length,truncation=True)
		pad_num = max_length-len(tokens_org)
		if pad_num>0:
			tokens_org.extend([0 for _ in range(pad_num)])
		mask_org = [1 if tokenid!=0 else 0 for tokenid in tokens_org]
		batch_data[i]['mask_org'] = np.array(mask_org)
		batch_data[i]['tokens_org'] = np.array(tokens_org)

	keys = list(batch_data[0].keys())
	batch_data_collated = {}
	for k in keys:
		batch_data_collated[k] = [data[k] for data in batch_data]

	for k in batch_data_collated:
		if isinstance(batch_data_collated[k][0], np.ndarray):
			batch_data_collated[k] = torch.tensor(batch_data_collated[k])
		elif isinstance(batch_data_collated[k][0], int) or isinstance(batch_data_collated[k][0], float)\
			or isinstance(batch_data_collated[k][0], np.int32) or isinstance(batch_data_collated[k][0], np.float32):
			batch_data_collated[k] = torch.tensor(batch_data_collated[k])

	return batch_data_collated

def kbert_two_stage_collate_fn_2item(data_list):
	data_list1 = [datum[0] for datum in data_list]
	data_list2 = [datum[1] for datum in data_list]
	return kbert_two_stage_collate_fn(data_list1), kbert_two_stage_collate_fn(data_list2)


collate_factory_train={
	'sentim': None,
	'causal': None,
	'base_DA': None,
	'kbert_two_stage_sentim': kbert_two_stage_collate_fn,
	'kbert_two_stage_da': kbert_two_stage_collate_fn_2item,
	'DANN_kbert': None
}

collate_factory_eval={
	'sentim': None,
	'causal': None,
	'base_DA': None,
	'kbert_two_stage_sentim': kbert_two_stage_collate_fn,
	'kbert_two_stage_da': kbert_two_stage_collate_fn,
	'DANN_kbert': None
}