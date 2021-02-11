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
# CUDA_VISIBLE_DEVICES='1,2,3,7' python train.py \
#     --config_path ./models/google_config.json \
#     --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
#     --dataset imdb \
#     --model_name sentim \
#     --task sentim \
#     --epochs_num 10 --batch_size 128 --kg_path data/imdb_sub_conceptnet.spo \
#     --log_dir log/sentim_1113 \
#     --num_gpus 4 \
#     --max_seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --print_freq 100 \
#     --use_kg
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

#baseDA
# CUDA_VISIBLE_DEVICES='5,6,7' python train.py \
#     --source bdek.electronics \
#     --target airlines \
#     --model_name base_DA \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/EA_base_0127 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --weight_decay 0.0001

#baseDA roberta

# CUDA_VISIBLE_DEVICES='2,3,4,5' python train.py \
#     --source bdek.electronics \
#     --target bdek.dvd \
#     --model_name base_DA_roberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/ED_base_r_0209 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --weight_decay 0.0001

# SSL DA
# CUDA_VISIBLE_DEVICES='3,5,6' python train.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name SSL_kbert \
#     --task DA_SSL \
#     --epochs_num 10 --batch_size 24 \
#     --log_dir log/DA_SSL_BE_0124_wd1e-4_02 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --num_pivots 500 \
#     --ssl_warmup 0.5 \
#     --weight_decay 0.0001

# CUDA_VISIBLE_DEVICES='0,1,2' python train.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name SSL_kbert \
#     --task DA_SSL \
#     --epochs_num 10 --batch_size 12 \
#     --log_dir log/DA_SSL_P_BE_0117 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --num_pivots 500 \
#     --ssl_warmup 0.5 \
#     --use_kg \
#     --kg_path 'data/results/electronics_unlabeled_org' \
#     --update 
    # --config_path ./models/google_config.json \
    # --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
#Pbert
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,6' python train.py \
#     --source bdek.dvd \
#     --target bdek.books \
#     --model_name base_DA \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --kg_path data/results/db_org \
#     --log_dir log/AB_Pbert_0210_b1 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --use_kg \
#     --balanced_interval 1

    # --kg_path data/amazon-review-old/books/unlabeled_graph.spo \

# P-roberta
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name base_DA_roberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 64 \
#     --kg_path data/results/be_org \
#     --log_dir log/BE_Pbert_r_0210_b1_f0.39 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --min_occur 10 \
#     --balanced_interval 1 \
#     --filter conf \
#     --filter_conf 0.18

# Pbert + SSL
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
#     --source airlines \
#     --target bdek.dvd \
#     --model_name masked_SSL_kbert \
#     --task masked_DA_SSL \
#     --epochs_num 10 --batch_size 32 \
#     --kg_path data/results/ad_org \
#     --log_dir log/AD_PbertSSL_0204_wd1e-4_b9_l0.28w0.1 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --ssl_warmup 0.1 \
#     --lambda_ssl 0.28 \
#     --use_kg \
#     --balanced_interval 9 \
#     --weight_decay 2e-4 \
#     --filter conf \
#     --filter_conf 0.1 \
#     --update \
#     --update_rate 2e-4

# CUDA_VISIBLE_DEVICES='8,9,10,11,12,13,14,15' python train.py \
#     --source airlines \
#     --target bdek.kitchen \
#     --model_name masked_SSL_kbert \
#     --task masked_DA_SSL \
#     --epochs_num 10 --batch_size 32 \
#     --kg_path data/results/ak_org \
#     --log_dir log/AK_PbertSSL_0203_wd1e-4_b9 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --ssl_warmup 0.5 \
#     --lambda_ssl 0.1 \
#     --use_kg \
#     --balanced_interval 9 \
#     --weight_decay 1e-4 \
#     --filter conf \
#     --filter_conf 0.1 \
#     --update \
#     --update_rate 2e-4

# masked ssl proberta
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
    --source bdek.electronics \
    --target bdek.books \
    --model_name masked_SSL_kroberta \
    --task masked_DA_SSL \
    --epochs_num 10 --batch_size 32 \
    --kg_path data/results/eb_org \
    --log_dir log/EB_PbertSSL_r_0211_b5_u2e-4_l0.3w0.1_wd1e-4_f0.40_lr2e-5 \
    --seq_length 256 \
    --learning_rate 2e-5 \
    --pooling first \
    --num_workers 32 \
    --print_freq 100 \
    --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
    --min_occur 10 \
    --ssl_warmup 0.1 \
    --lambda_ssl 0.3 \
    --use_kg \
    --balanced_interval 5 \
    --filter_conf 0.40 \
    --filter conf \
    --weight_decay 0.0001 \
    --update \
    --update_rate 0.0002 \
    --update_steps 10

# Pbert eval
# CUDA_VISIBLE_DEVICES='2,5' python evaluate.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name base_DA \
#     --task domain_adaptation \
#     --epochs_num 100 --batch_size 16 \
#     --kg_path data/results/electronics_unlabeled_org \
#     --log_dir log/DA_Pbert_0123_01 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --trained_model_path ./log/BE_Pbert_0122_balanced/model_best.pth \
#     --min_occur 10 \
    # --use_kg \
    # --save_attention_mask

# SSL+DANN
# CUDA_VISIBLE_DEVICES='2,3,4,5' python3 train.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name SSL_kbert_DANN \
#     --task DA_SSL \
#     --epochs_num 10 --batch_size 32 \
#     --kg_path data/results/electronics_unlabeled_org \
#     --log_dir log/DA_Pbert_DANN_0117 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --config_path ./models/google_config.json \
#     --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
#     --use_kg \
#     --gamma 0.5

# DANN kbert
# CUDA_VISIBLE_DEVICES='4,5' python train.py \
#     --source bdek.electronics\
#     --target bdek.dvd \
#     --model_name DANN_kbert \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/ED_DANN_0127_g0.25 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --gamma 0.25

# DANN kroberta
# CUDA_VISIBLE_DEVICES='4,5,6,7' python train.py \
#     --source bdek.kitchen \
#     --target bdek.dvd \
#     --model_name DANN_kroberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/KD_DANN_R_0209_g0.15 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --min_occur 10 \
#     --gamma 0.15

# CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py \
#     --source bdek.kitchen \
#     --target bdek.electronics \
#     --model_name DANN_kroberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/KE_DANN_R_0209_g0.25 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --min_occur 10 \
#     --gamma 0.25
