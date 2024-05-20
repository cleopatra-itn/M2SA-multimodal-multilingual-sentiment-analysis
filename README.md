# Multilingual-Multimodal-Sentiment-Analysis

 
This is the official Git repository page for the paper:

```
Gaurish Thakkar, Sherzod Hakimov and Marko Tadić. 2024. 
"M2SA: Multimodal and Multilingual Model for Sentiment Analysis of Tweets". 
The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), Torino (Italia)

```

- [Paper](https://arxiv.org/abs/2404.01753)
- [Dataset](final_dataset.zip) with tweetid and label only
- Models [multimodal](https://huggingface.co/FFZG-cleopatra/M2SA), [text-only](https://huggingface.co/FFZG-cleopatra/M2SA-text-only)
- Demo [multimodal](https://huggingface.co/spaces/FFZG-cleopatra/M2SA-demo-multimodal), [text-only](https://huggingface.co/spaces/FFZG-cleopatra/M2SA-demo-text-only)

For access to images and tweets, send an email with organization (university/institute) and purpose/usage details to gthakkar@m.ffzg.hr

## Environment Setup
### Installation
Create a virtual environment in Python 3.10 and run the following scripts

```
pip install -r requirements.txt

```

## Data Setup
### 1. Twitter text-image preparation/curation

```
python twitter_curation/curate.py
```

### 2. Language translation using https://huggingface.co/facebook/nllb-200-3.3B model

```
python twitter_curation/translation.py
```

## Training

0. Run the following command to create vision text dual encoder model (clip-mbert)

```
python src_VTDM/prepare.py
```
1.1 To train model with DINOv2 and mBERT run

```
# Define an array of seeds
SEEDS=(42 123 777 2020 31337)

LANGUAGES=("ar"  "bg" "bs" "da" "de" "en" "es" "fr" "hr" "hu" "it" "mt" "pl" "pt" "ru" "sr" "sv" "tr" "zh" "lv" "sq" "bg_mt" "bs_mt" "da_mt" "fr_mt" "hr_mt" "hu_mt"  "mt_mt" "pt_mt" "ru_mt"  "sr_mt" "sv_mt" "tr_mt" "zh_mt")


# Use a for loop to iterate over the alphabets
for LANG in "${LANGUAGES[@]}"
do
    echo "Language: $LANG"
    for seed in "${SEEDS[@]}"
    do
        echo "Seed: $seed"
        cuda-wrapper.sh  python src_VTDEM/train-dino-mbert.py \
                --output_dir ./mbert-dino-finetuned-"$LANG"-"$seed"  \
                --model_name_or_path                bert-base-multilingual-uncased \
                --image_processor_name facebook/dinov2-base \
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
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --auto_find_batch_size \
                --learning_rate 5e-5 \
                --warmup_steps 0 \
                --weight_decay 0.1 \
                --overwrite_output_dir yes \
                --seed "$seed" \
                --load_best_model_at_end yes \
                --evaluation_strategy   steps \
                --fp16  \
                --optim adafactor \
                --disable_tqdm="True" \
                --save_total_limit 2

    done

done
```
To train with CLIP-mBERT run

```
SEEDS=(42 123 777 2020 31337)

LANGUAGES=("ar"  "bg" "bs" "da" "de" "en" "es" "fr" "hr" "hu" "it" "mt" "pl" "pt" "ru" "sr" "sv" "tr" "zh" "lv" "sq" "bg_mt" "bs_mt" "da_mt" "fr_mt" "hr_mt" "hu_mt"  "mt_mt" "pt_mt" "ru_mt"  "sr_mt" "sv_mt" "tr_mt" "zh_mt")

for LANG in "${LANGUAGES[@]}"
do
    echo "Language: $LANG"
    for seed in "${SEEDS[@]}"
    do
        echo "Seed: $seed"
        cuda-wrapper.sh  python src_VTDEM/train.py \
                --output_dir ./clip-mbert-finetuned-"$LANG"-"$seed"  \
                --model_name_or_path                ./clip-mbert \
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
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --auto_find_batch_size \
                --learning_rate 5e-5 \
                --warmup_steps 0 \
                --weight_decay 0.1 \
                --overwrite_output_dir yes \
                --seed "$seed" \
                --load_best_model_at_end yes \
                --evaluation_strategy   steps \
                --fp16  \
                --optim adafactor 
    done
done

```



2. To train text only models run

```

# Define an array of seeds
SEEDS=(42 123 777 2020 31337)

LANGUAGES=("ar"  "bg" "bs" "da" "de" "en" "es" "fr" "hr" "hu" "it" "mt" "pl" "pt" "ru" "sr" "sv" "tr" "zh" "lv" "sq" "bg_mt" "bs_mt" "da_mt" "fr_mt" "hr_mt" "hu_mt"  "mt_mt" "pt_mt" "ru_mt"  "sr_mt" "sv_mt" "tr_mt" "zh_mt")

# Use a for loop to iterate over the alphabets
for LANG in "${LANGUAGES[@]}"
do
    echo "Language: $LANG"
    for seed in "${SEEDS[@]}"
    do
        echo "Seed: $seed"
        cuda-wrapper.sh  python src_VTDEM/train-text-only.py \
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
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --auto_find_batch_size \
                --learning_rate 5e-5 \
                --warmup_steps 0 \
                --weight_decay 0.1 \
                --overwrite_output_dir yes \
                --seed "$seed" \
                --load_best_model_at_end yes \
                --evaluation_strategy   steps \
                --fp16  \
                --optim adafactor \
                --disable_tqdm="True" \
                --save_total_limit 2

    done

done

```
3. To train the Twitter-XLMR and CLIP model run

```
# Define an array of seeds
SEEDS=(42 123 777 2020 31337)

LANGUAGES=("ar" "bg" "bs" "da" "de" "en" "es" "fr" "hr" "hu" "it" "mt" "pl" "pt" "ru" "sr" "sv" "tr" "zh" "lv" "sq" "bg_mt" "bs_mt" "da_mt" "fr_mt" "hr_mt" "hu_mt"  "mt_mt" "pt_mt" "ru_mt"  "sr_mt" "sv_mt" "tr_mt" "zh_mt")




# Use a for loop to iterate over the alphabets
for LANG in "${LANGUAGES[@]}"
do
    echo "Language: $LANG"
    for seed in "${SEEDS[@]}"
    do
        echo "Seed: $seed"
        cuda-wrapper.sh  python src_VTDEM/train-twitter-xlmr-clip.py \
                --output_dir ./twitter-xlmr-clip-finetuned-"$LANG"-"$seed"  \
                --model_name_or_path                cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \
                --image_processor_name openai/clip-vit-base-patch32 \
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
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --auto_find_batch_size \
                --learning_rate 5e-5 \
                --warmup_steps 0 \
                --weight_decay 0.1 \
                --overwrite_output_dir yes \
                --seed "$seed" \
                --load_best_model_at_end yes \
                --evaluation_strategy   steps \
                --fp16  \
                --optim adafactor \
                --disable_tqdm="True" \
                --save_total_limit 2

    done

done


```

## Evaluation

To eval a model run the following code
```
conda activate environment

# Define an array of seeds # 123 777 2020 31337
SEEDS=(42 123 777 2020 31337)

LANGUAGES=("ar"  "bg" "bs" "da" "de" "en" "es" "fr" "hr" "hu" "it" "mt" "pl" "pt" "ru" "sr" "sv" "tr" "zh" "lv" "sq" "bg_mt" "bs_mt" "da_mt" "fr_mt" "hr_mt" "hu_mt"  "mt_mt" "pt_mt" "ru_mt"  "sr_mt" "sv_mt" "tr_mt" "zh_mt")

# Use a for loop to iterate over the alphabets
for LANG in "${LANGUAGES[@]}"
do
    echo Language: $LANG
    for seed in ${SEEDS[@]}
    do
        echo Seed: $seed
        python src_VTDEM/eval.py \
                --output_dir ./twitter-xlmr-txtonly-finetuned-evaluation-all-"$seed"/"$LANG" \
                --model_name_or_path cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \
                --seed "$seed" \
                --image_processor_name openai/clip-vit-base-patch32 \
                --dataset_name "$LANG" \
                --data_dir ./data \
                --image_column  image_paths \
                --caption_column normalized_text \
                --label_column label \
                --remove_unused_columns False \
                --do_eval \
                --do_predict \
                --per_device_eval_batch_size 128\
                --overwrite_output_dir yes \
                --evaluation_strategy steps \
                --fp16 \
                --optim adafactor 

       

    done

done


