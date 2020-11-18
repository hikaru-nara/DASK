from kbert_model import BertModel
from model_config import BertConfig
import torch
config = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel(config=config, add_pooling_layer=False)

inputs = torch.arange(8*128).reshape(8,-1).long()
masks = torch.ones(8,128)

pretrain_path = '../models/pytorch-bert-uncased/pytorch_model.bin'
state_dict = torch.load(pretrain_path)
state_dict_align = {}
for k,v in state_dict.items():
	spk = k.split('.')
	if spk[0]=='bert':
		
		if spk[-2]=='LayerNorm':
			spk[-1] = 'weight' if spk[-1]=='gamma' else 'bias'

		k = '.'.join(spk[1:])
		state_dict_align[k]=v
	else:
		state_dict_align[k]=v

model.load_state_dict(state_dict_align, strict=False)



output = model(inputs, masks)[0]
print(output.shape)
# embedding = model.embeddings(inputs, mask)
# output = model.encoder(embedding, mask)