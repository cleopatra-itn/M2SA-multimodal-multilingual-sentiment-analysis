# /home/gaurish/miniconda3/envs/venv/bin/python
# 1. Run prepare.py


conda activate py310

# Define an array of seeds
# SEEDS=(42 123 777 2020 31337)

LANGUAGES="all"

# Use a for loop to iterate over the alphabets
for LANG in $LANGUAGES
do
    echo "Language: $LANG"
    for seed in "${SEEDS[@]}"
    do
        echo "Seed: $seed"
        # python src_VTDEM/train.py \
        #         --output_dir ./clip-mbert-finetuned-"$LANG"-"$seed"  \
        #         --model_name_or_path                ./clip-mbert \
        # python src_VTDEM/train-twitter-xlmr-clip.py \
        #         --output_dir ./twitter-xlmr-clip-finetuned-"$LANG"-"$seed"  \
        #         --model_name_or_path    cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \
        #         --image_processor_name openai/clip-vit-base-patch32 \
        python src_VTDEM/train-text-only.py \
                --output_dir ./cardiffnlp-twitter-xlmr-finetuned-txtnly-"$LANG"-"$seed"  \
                --model_name_or_path   cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \
                --dataset_name   "$LANG" \
                --data_dir   ./data \
                --image_column                image_paths \
                --caption_column                normalized_text \
                --label_column                label \
                --remove_unused_columns                False \
                --do_train \
                --do_eval \
                --do_predict \
                --num_train_epochs="50" \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --auto_find_batch_size \
                --learning_rate 5e-5 \
                --warmup_steps 0 \
                --weight_decay 0.1 \
                --overwrite_output_dir yes \
                --seed "$seed" \
                --load_best_model_at_end yes \
                --evaluation_strategy   steps \
                --fp16  \
                --save_total_limit 3 \
                --optim adafactor 

    done

done


