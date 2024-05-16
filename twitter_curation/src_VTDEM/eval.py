from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from datasets import load_metric

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict, disable_caching

disable_caching()

from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import evaluate


from model import ClassificationHead, VisionTextClassificationModel
from utils import (
    ModelArguments,
    DataTrainingArguments,
    dataset_name_mapping,
    Transform,
    CustomTrainer,
    collate_fn,
)

logger = logging.getLogger(__name__)


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/evaluation_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, evaluation_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, evaluation_args = parser.parse_args_into_dataclasses()

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

    if evaluation_args.should_log:
        # The default of evaluation_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = evaluation_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {evaluation_args.local_rank}, device: {evaluation_args.device}, n_gpu: {evaluation_args.n_gpu}"
        + f"distributed training: {evaluation_args.parallel_mode.value == 'distributed'}, 16-bits training: {evaluation_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {evaluation_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(evaluation_args.output_dir)
        and evaluation_args.do_train
        and not evaluation_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(evaluation_args.output_dir)
        if last_checkpoint is None and len(os.listdir(evaluation_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({evaluation_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and evaluation_args.resume_from_checkpoint is None
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

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # Load the intial data
        dataset_init = load_dataset(
            "json",
            data_files=data_args.dataset_name
        )

        stratify_column_name = data_args.label_column  # "label"

        dataset_init = dataset_init.class_encode_column(stratify_column_name)

        # train validation test split
        train_testvalid = dataset_init["train"].train_test_split(
            test_size=0.1,
            stratify_by_column=stratify_column_name,
            seed=evaluation_args.seed,
        )

        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid["test"].train_test_split(
            test_size=0.5,
            stratify_by_column=stratify_column_name,
            seed=evaluation_args.seed,
        )

        # gather everyone if you want to have a single DatasetDict
        dataset = DatasetDict(
            {
                # "train": train_testvalid["train"],
                # "validation": test_valid["train"],
                "test": test_valid["test"],
            }
        )
        #

    else:
        data_files = {}
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

    model.load_state_dict(torch.load(model_args.model_name_or_path+"-finetuned/pytorch_model.bin"))

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(evaluation_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if evaluation_args.do_train:
        print("Shouldnt run this.")
    elif evaluation_args.do_eval:
        column_names = dataset["test"].column_names
    elif evaluation_args.do_predict:
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
        examples["pixel_values"] = [image_transformations(image) for image in images]
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

    # TODO https://huggingface.co/spaces/ludvigolsen/plot_confusion_matrix
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

    

    if evaluation_args.do_eval:
        if "test" not in dataset:
            raise ValueError("--do_eval requires a test/ validation")
        eval_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
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
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    if evaluation_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
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
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    # 8. Initalize our trainer
    trainer = CustomTrainer(
        model=model,
        args=evaluation_args,
        eval_dataset=eval_dataset if evaluation_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # 9. Training

    # 10. Evaluation
    if evaluation_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

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

    print(kwargs)


if __name__ == "__main__":
    main()
