# https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/README.md
#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

from utils import (ModelArguments, DataTrainingArguments, dataset_name_mapping,
                   Transform, CustomTrainer, collate_fn, write_preds_to_file)
from dsstore import get_dataset_files
from model import (ClassificationHead, VisionTextClassificationModel)
import evaluate
from transformers.utils.versions import require_version
from transformers.utils import check_min_version, send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
import transformers
from torchvision.io import ImageReadMode, read_image
from PIL import Image
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict, disable_caching, concatenate_datasets, Value

disable_caching()


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.34.0.dev0")

# require_version(
#     "datasets>=1.8.0",
#     "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt",
# )
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #

    # Column name : "label"
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    # stratify only after everything is in the dataset object
    stratify_column_name = data_args.label_column

    return_with_translations = False

    if data_args.dataset_name.endswith("_mt"):
        return_with_translations = True

    if data_args.dataset_name is not None:
        # Loading a dataset files from the collection. returens {"train":[],"validation":[],"test":[]}
        data_files = get_dataset_files(
            data_args.dataset_name, return_with_translations)
        ds_translation = []
        if return_with_translations:
            ds_translation = load_dataset(
                "json", data_files=data_files["translations"])
            ds_translation = ds_translation.cast_column(
                'tweetid', Value(dtype='int64', id=None))
            ds_translation = ds_translation.class_encode_column(
                stratify_column_name)
            ds_translation = ds_translation.align_labels_with_mapping(
                label2id, stratify_column_name
            )

        if len(data_files["test"]) == 0 and len(data_files["validation"]) == 0:
            # fetch it from train set. Hoping train set is not null

            # Load the intial data with only train and derive validation and test
            dataset_init = load_dataset(
                "json",
                data_files=data_files["train"]
            )
            # TODO 1. Drop rows that are not in negative,neutral or positive.
            # TODO 2. Drop rows that are not in negative,neutral or positive.

            # TODO Test this
            dataset_init = dataset_init.cast_column(
                'tweetid', Value(dtype='int64', id=None))
            dataset_init = dataset_init.class_encode_column(
                stratify_column_name)
            dataset_init = dataset_init.align_labels_with_mapping(
                label2id, stratify_column_name
            )

            # train validation test split
            train_testvalid = dataset_init["train"].train_test_split(
                test_size=0.1,
                stratify_by_column=stratify_column_name,
                seed=training_args.seed,
            )

            # Split the 10% test + valid in half test, half valid
            test_valid = train_testvalid["test"].train_test_split(
                test_size=0.5,
                stratify_by_column=stratify_column_name,
                seed=training_args.seed,
            )

            # gather everyone if you want to have a single DatasetDict
            dataset = DatasetDict(
                {
                    "train": concatenate_datasets(
                        [train_testvalid["train"], ds_translation["train"]]
                    )
                    if return_with_translations
                    else train_testvalid["train"],
                    # train_testvalid["train"],
                    "test": test_valid["test"],
                    "validation": test_valid["train"],
                }
            )
        else:
            if len(data_files["validation"]) > 0 and len(data_files["test"]) == 0:
                dataset_init = load_dataset("json", data_files={"train": data_files["train"],
                                                                "validation": data_files["validation"],
                                                                "test": data_files["test"]})
                # gather everyone if you want to have a single DatasetDict
                dataset_init = dataset_init.cast_column(
                    'tweetid', Value(dtype='int64', id=None))
                dataset_init = dataset_init.class_encode_column(
                    stratify_column_name)
                dataset_init = dataset_init.align_labels_with_mapping(
                    label2id, stratify_column_name)

                # train validation test split
                train_test = dataset_init["train"].train_test_split(
                    test_size=0.1,
                    stratify_by_column=stratify_column_name,
                    seed=training_args.seed,
                )

                # gather everyone if you want to have a single DatasetDict
                dataset = DatasetDict(
                    {
                        "train": concatenate_datasets(
                            [train_test["train"], ds_translation["train"]]
                        )
                        if return_with_translations
                        else train_test["train"],
                        # "train": train_test["train"],
                        "test": train_test["test"],
                        "validation": dataset_init["valid"],
                    }
                )
            elif len(data_files["validation"]) == 0 and len(data_files["test"]) > 0:
                dataset_init = load_dataset("json", data_files={"train": data_files["train"],
                                                                "test": data_files["test"]})
                print(dataset_init)
                # gather everyone if you want to have a single DatasetDict
                dataset_init = dataset_init.cast_column(
                    'tweetid', Value(dtype='int64', id=None))
                dataset_init = dataset_init.class_encode_column(
                    stratify_column_name)
                dataset_init = dataset_init.align_labels_with_mapping(
                    label2id, stratify_column_name)

                # train validation test split
                train_test = dataset_init["train"].train_test_split(
                    test_size=0.1,
                    stratify_by_column=stratify_column_name,
                    seed=training_args.seed,
                )

                # gather everyone if you want to have a single DatasetDict
                dataset = DatasetDict(
                    {
                        "train": concatenate_datasets(
                            [train_test["train"], ds_translation["train"]]
                        )
                        if return_with_translations
                        else train_test["train"],
                        # "train": train_test["train"],
                        "validation": train_test["test"],
                        "test": dataset_init["test"],
                    }
                )
            else:
                dataset = load_dataset("json", data_files={"train": data_files["train"] + data_files["translations"],
                                                           "validation": data_files["validation"],
                                                           "test": data_files["test"]})
                dataset = dataset.cast_column(
                    'tweetid', Value(dtype='int64', id=None))
                dataset = dataset.class_encode_column(
                    stratify_column_name)
                dataset = dataset.align_labels_with_mapping(
                    label2id, stratify_column_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    backbone_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Create an instance of the combined model
    classification_head = ClassificationHead(1024, num_classes=3)

    model = VisionTextClassificationModel(backbone_model, classification_head)

    config = model.vision_text_model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    print(training_args.seed)
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        config.vision_config.image_size,
        image_processor.image_mean,
        image_processor.image_std,
    )
    image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(
            captions,
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        examples["labels"] = [label for label in examples["label"]]
        return examples

    def transform_images(examples):
        images = [
            read_image("./data/" + image_file[0], mode=ImageReadMode.RGB)
            for image_file in examples[image_column]
        ]
        examples["pixel_values"] = [
            image_transformations(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open("./data/" + image_file[0])  # take the first one
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    def compute_metrics(eval_pred):
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["recall"]
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"]
        return {"precision": precision, "recall": recall, "f1": f1}

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[
                col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[
                col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))

        test_dataset = test_dataset.filter(
            filter_corrupt_images,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        test_dataset = test_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[
                col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    # 8. Initalize our trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["seed"] = training_args.seed
        metrics["train_size"] = len(train_dataset)
        metrics["valid_size"] = len(eval_dataset)
        metrics["test_size"] = len(test_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        prediction_logits, original_labels, _ = trainer.predict(
            test_dataset, metric_key_prefix="test")

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        predictions = np.argmax(prediction_logits, axis=-1)
        texts = tokenizer.batch_decode(
            test_dataset[:]["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        write_preds_to_file(texts, predictions,
                            original_labels, training_args.output_dir)

    # 11. Write Training Stats and push to hub.
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "contrastive-image-text-modeling",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
