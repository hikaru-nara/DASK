# graph-causal-domain-adaptation

## requirements

```
pip install requirements.txt
```

## Prepare

* Download Bert-base-uncased pretrain weights from [here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz), or see a list of Bert model weights download links [here](https://github.com/maknotavailable/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py)
* Download corresponding vocabulary [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip). Note that the downloaded tar also contains the tensorflow pretrained model weights, but we only need the file ``vocab.txt`` 
* Put the pretrained model file, the config json downloaded from the first step, and the vocabulary to ``models/pytorch-bert-uncased`` directory. 
* Download imdb dataset [here](http://ai.stanford.edu/~amaas/data/sentiment/) and put it to ``data/imdb``
* Download bdek dataset(i.e. amazon reviews dataset) [here](https://github.com/yftah89/Neural-SCL-Domain-Adaptation/tree/master/data) and put it to ``data/bdek``

## Run

* run ``sh train_script.sh`` in shell
  * open this file and you'll see different commands for different tasks

## To develop

* The start point of the program is ``train.py``
* Files like ``trainers.py, evaluators.py, model.py, dataset.py ``, etc., defines classes for the corresponding component of the program, and is imported to ``train.py`` by xx_factory at the bottom of each file.
* Developer should add new classes to these files to implement new features instead of editting the existing ones.
* There are several command line args that effect which module to choose from the factories, see the code for details.

## Kindly Note

This Readme may not have covered everything. If anything is unclear, contact me on wechat. Thanks for your collaboration.



