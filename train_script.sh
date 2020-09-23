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

# sentim
# CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py \
#     --config_path ./models/google_config.json \
#     --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
#     --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
#     --dataset imdb \
#     --model_name sentim \
#     --task sentiment \
#     --epochs_num 100 --batch_size 64 --kg_path data/imdb_sub_conceptnet.spo \
#     --log_dir log/sentim_0923 \
#     --num_gpus 4 \
#     --seq_length 256 \
#     --learning_rate 1e-5 \
#     --num_workers 16 \
#     --print_freq 100

# # causal
CUDA_VISIBLE_DEVICES='4,5,6,7' python3 -u train.py \
    --config_path ./models/google_config.json \
    --vocab_path ./models/pytorch-bert-uncased/vocab.txt \
    --pretrained_model_path ./models/pytorch-bert-uncased/pytorch_model.bin \
    --dataset imdb \
    --model_name causal \
    --task causal_inference \
    --epochs_num 100 --batch_size 64 \
    --kg_path data/imdb_sub_conceptnet.spo \
    --log_dir log/causal_0923 \
    --num_gpus 4 \
    --seq_length 256 \
    --optimizer adam \
    --learning_rate 1e-5 \
    --num_workers 16 \
#     --print_freq 50


    # --pretrained_model_path ./models/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    # --config_path ./models/multi_cased_L-12_H-768_A-12/bert_config.json \
    # --vocab_path ./models/multi_cased_L-12_H-768_A-12/vocab.txt \