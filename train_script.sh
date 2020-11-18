# DA
# python3 -u train.py \
#     --pretrained_model_path ./models/google_model.bin \
#     --config_path ./models/google_config.json \
#     --vocab_path ./models/google_vocab.txt \
#     --source bdek.books \
#     --target bdek.dvd \
#     --epochs_num 100 --batch_size 32 --kg_path data/bdek_sub_conceptnet.spo \
#     --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin \
#     --log_dir log/first_test \
#     --gpus 13\

#sentim
CUDA_VISIBLE_DEVICES='1,2,3,7' python train.py \
    --config_path ./models/google_config.json \
    --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
    --dataset imdb \
    --model_name sentim \
    --task sentim \
    --epochs_num 10 --batch_size 128 --kg_path data/imdb_sub_conceptnet.spo \
    --log_dir log/sentim_1113 \
    --num_gpus 4 \
    --max_seq_length 256 \
    --learning_rate 2e-5 \
    --pooling first \
    --num_workers 24 \
    --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
    --print_freq 100 \
    --use_kg
#--pooling max \
# # causal
# --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
    
# CUDA_VISIBLE_DEVICES='0,1,10,11' python3 -u train.py \
#     --config_path ./models/google_config.json \
#     --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --dataset imdb \
#     --model_name causal \
#     --task causal_inference \
#     --epochs_num 100 --batch_size 32 \
#     --kg_path data/imdb_sub_conceptnet.spo \
#     --log_dir log/causal_0926_2 \
#     --num_gpus 4 \
#     --seq_length 256 \
#     --optimizer adam \
#     --learning_rate 2e-5 \
#     --num_workers 16 \
#     --init normal \
#     --freeze_bert \
#     --print_freq 50


    # --pretrained_model_path ./models/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    # --config_path ./models/multi_cased_L-12_H-768_A-12/bert_config.json \
    # --vocab_path ./models/multi_cased_L-12_H-768_A-12/vocab.txt \