python sp_model/generate_sp.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --cache_dir cache \
    --train_file data/train.json \
    --test_file data/test.json \
    --output_dir models/extract-and-build \
    --overwrite_output_dir \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --predict_with_generate \
    --num_beams 4 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --learning_rate 3e-5 \
    --save_steps 20000 \
    --max_source_length 512 \
    --max_target_length 100 \
    --num_train_epochs 6 \