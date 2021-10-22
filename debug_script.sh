#Pbert
CUDA_VISIBLE_DEVICES='0' python train.py \
    --debug \
    --source bdek.books \
    --target bdek.electronics \
    --model_name base_DA \
    --task domain_adaptation \
    --epochs_num 10 --batch_size 1 \
    --kg_path data/results/be_org \
    --log_dir log/BE_Pbert_1019_b1 \
    --seq_length 16 \
    --learning_rate 2e-5 \
    --pooling first \
    --num_workers 24 \
    --print_freq 100 \
    --pretrained_model_path ./models/bert-base-uncased/pytorch_model.bin \
    --min_occur 10 \
    --use_kg \
    --balanced_interval 1 \
    --filter conf_uncased \
    --filter_conf 0.0 \
    --refilter

