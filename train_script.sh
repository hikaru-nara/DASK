python3 -u train.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --source books \
    --target dvd \
    --epochs_num 100 --batch_size 32 --kg_path data/bdek_sub_conceptnet.spo \
    --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin \
    --log_dir log/first_test \
    --gpus 13\