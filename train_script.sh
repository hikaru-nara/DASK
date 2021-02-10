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
 


    # --pretrained_model_path ./models/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    # --config_path ./models/multi_cased_L-12_H-768_A-12/bert_config.json \
    # --vocab_path ./models/multi_cased_L-12_H-768_A-12/vocab.txt \

# SSL DA
# CUDA_VISIBLE_DEVICES='0,5,6,7' python3 train.py \
#     --source airlines \
#     --target bdek.books \
#     --model_name SSL_kbert \
#     --task DA_SSL \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/AB_SSL_0204_l0.25w0.1_nomask \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --num_pivots 500 \
#     --ssl_warmup 0.1 \
#     --lambda_ssl 0.25 
    # --config_path ./models/google_config.json \
    # --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
#Pbert
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
#     --source bdek.electronics \
#     --target bdek.books \
#     --model_name base_DA_roberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 64 \
#     --kg_path data/results/eb_org \
#     --log_dir log/EB_Pbert_r_0209_b9 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --min_occur 10 \
#     --balanced_interval 9 \
#     --filter conf \
#     --filter_conf 0.40

#baseDA
# CUDA_VISIBLE_DEVICES='6' python train.py \
#     --source bdek.electronics \
#     --target bdek.books \
#     --model_name base_DA \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/EB_base_0127 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin 


# baseDA roberta
# CUDA_VISIBLE_DEVICES='1,3,4,5' python train.py \
#     --source bdek.books \
#     --target bdek.kitchen \
#     --model_name base_DA_roberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 64 \
#     --log_dir log/DE_base_r_0209 \
#     --seq_length 256 \
#     --learning_rate 2e-6 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --weight_decay 0.0001 \
#     --warmup 0.2

# DANN bert
# CUDA_VISIBLE_DEVICES='6' python train.py \
#     --source bdek.kitchen \
#     --target airlines\
#     --model_name DANN_kbert \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 32 \
#     --log_dir log/KA_DANN_0131_wd1e-2_g0.25 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --gamma 0.25 \
#     --weight_decay 0.01

# DANN roberta
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
#     --source bdek.electronics \
#     --target bdek.dvd \
#     --model_name DANN_kroberta \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 64 \
#     --log_dir log/ED_DANN_r_0209_g0.25 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
#     --min_occur 10 \
#     --gamma 0.25 \
#     --weight_decay 0.0001


# Pbert + SSL
# CUDA_VISIBLE_DEVICES='0,5,6,7' python train.py \
#     --source airlines \
#     --target bdek.books \
#     --model_name SSL_kbert \
#     --task DA_SSL \
#     --epochs_num 10 --batch_size 64 \
#     --kg_path data/results/ab_org \
#     --log_dir log/BE_PbertSSL_0203_nou_b9_l0.25w0.1_wd1e-4 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --ssl_warmup 0.1 \
#     --lambda_ssl 0.25 \
#     --use_kg \
#     --balanced_interval 9 \
#     --filter_conf 0.1 \
#     --filter conf 

# masked ssl pbert
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
#     --source airlines \
#     --target bdek.dvd \
#     --model_name masked_SSL_kbert \
#     --task masked_DA_SSL \
#     --epochs_num 10 --batch_size 64 \
#     --kg_path data/results/ad_org \
#     --log_dir log/AD_PbertSSL_0205_b9_u2e-4_l0.2w0.1_wd2e-4 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 32 \
#     --print_freq 100 \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --min_occur 10 \
#     --ssl_warmup 0.1 \
#     --lambda_ssl 0.2 \
#     --use_kg \
#     --balanced_interval 9 \
#     --filter_conf 0.18 \
#     --filter conf \
#     --weight_decay 0.0002 \
#     --update \
#     --update_rate 0.0002 \
#     --update_steps 10

# masked ssl proberta
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python train.py \
    --source bdek.electronics \
    --target bdek.books \
    --model_name masked_SSL_kroberta \
    --task masked_DA_SSL \
    --epochs_num 10 --batch_size 64 \
    --kg_path data/results/eb_org \
    --log_dir log/EB_PbertSSL_r_0210_b9_nu_l0.1w0.1_wd1e-4_f0.40_lr2e-5 \
    --seq_length 256 \
    --learning_rate 2e-5 \
    --pooling first \
    --num_workers 32 \
    --print_freq 100 \
    --pretrained_model_path ./models/pytorch-roberta-base/pytorch_model.bin \
    --min_occur 10 \
    --ssl_warmup 0.1 \
    --lambda_ssl 0.1 \
    --use_kg \
    --balanced_interval 9 \
    --filter_conf 0.40 \
    --filter conf \
    --weight_decay 0.0001 \
    --update \
    --update_rate 0.0002 \
    --update_steps 10

#Pbert eval
# CUDA_VISIBLE_DEVICES='0,1,2' python3 evaluate.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name base_DA \
#     --task domain_adaptation \
#     --epochs_num 10 --batch_size 48 \
#     --kg_path data/results/electronics_unlabeled_org \
#     --log_dir log/DA_Pbert_0117 \
#     --seq_length 256 \
#     --learning_rate 2e-5 \
#     --pooling first \
#     --num_workers 24 \
#     --print_freq 100 \
#     --trained_model_path ./log/DA_Pbert_0116/model_best.pth \
#     --min_occur 10 \
#     --use_kg \
#     --save_attention_mask

# SSL+DANN
# CUDA_VISIBLE_DEVICES='2,3,4,5' python3 train.py \
#     --source bdek.books \
#     --target bdek.electronics \
#     --model_name SSL_kbert_DANN \
#     --task DA_SSL \
#     --epochs_num 4 --batch_size 32 \
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
#     --gamma 0.5 \
#     --lambda_ssl 0.2 \
#     --ssl_warmup 0

