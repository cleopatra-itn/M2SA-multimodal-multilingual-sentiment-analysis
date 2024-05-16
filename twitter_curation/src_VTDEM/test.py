import dsstore as store
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict, disable_caching, concatenate_datasets, Value

disable_caching()


# print(store.get_dataset_files("ar"))

for key in store.DATASET_STORE.keys():

    print(key)
    print(store.get_dataset_files(key),key.endswith("_mt"))
    print()
    # Column name : "label"
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    # stratify only after everything is in the dataset object
    stratify_column_name = "label"
    dataset_name = key

    return_with_translations = False

    if dataset_name.endswith("_mt"):
        return_with_translations = True

    # Loading a dataset files from the collection. returens {"train":[],"validation":[],"test":[]}
    data_files = store.get_dataset_files(dataset_name, return_with_translations)
    ds_translation = []
    if return_with_translations:
        ds_translation = load_dataset("json", data_files=data_files["translations"])
        ds_translation = ds_translation.cast_column('tweetid', Value(dtype='int64', id=None))
        ds_translation = ds_translation.class_encode_column(stratify_column_name)
        ds_translation = ds_translation.align_labels_with_mapping(
            label2id, stratify_column_name
        )

    if len(data_files["test"]) == 0 and len(data_files["validation"]) == 0:
        # fetch it from train set. Hoping train set is not null

        # Load the intial data with only train and derive validation and test
        dataset_init = load_dataset("json", data_files=data_files["train"])

        # TODO 1. Drop rows that are not in negative,neutral or positive.
        dataset_init = dataset_init.cast_column('tweetid', Value(dtype='int64', id=None))
        dataset_init = dataset_init.class_encode_column(stratify_column_name)
        dataset_init = dataset_init.align_labels_with_mapping(
            label2id, stratify_column_name
        )

        # train validation test split
        train_testvalid = dataset_init["train"].train_test_split(
            test_size=0.1,
            stratify_by_column=stratify_column_name,
            seed=42,
        )

        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid["test"].train_test_split(
            test_size=0.5,
            stratify_by_column=stratify_column_name,
            seed=42,
        )

        # gather everyone if you want to have a single DatasetDict
        dataset = DatasetDict(
            {
                "train": concatenate_datasets(
                    [train_testvalid["train"], ds_translation["train"]]
                )
                if return_with_translations
                else train_testvalid["train"],
                "test": test_valid["test"],
                "validation": test_valid["train"],
            }
        )
    else:
        if len(data_files["validation"]) > 0 and len(data_files["test"]) == 0:
            dataset_init = load_dataset(
                "json",
                data_files={
                    "train": data_files["train"],
                    "validation": data_files["validation"],
                    "test": data_files["test"],
                },
            )
            # gather everyone if you want to have a single DatasetDict
            dataset_init = dataset_init.cast_column('tweetid', Value(dtype='int64', id=None))
            dataset_init = dataset_init.class_encode_column(stratify_column_name)
            dataset_init = dataset_init.align_labels_with_mapping(
                label2id, stratify_column_name
            )

            # train validation test split
            train_test = dataset_init["train"].train_test_split(
                test_size=0.1,
                stratify_by_column=stratify_column_name,
                seed=42,
            )

            # gather everyone if you want to have a single DatasetDict
            dataset = DatasetDict(
                {
                    "train": concatenate_datasets(
                        [train_test["train"], ds_translation["train"]]
                    )
                    if return_with_translations
                    else train_test["train"],
                    "test": train_test["test"],
                    "validation": dataset_init["valid"],
                }
            )
        elif len(data_files["validation"]) == 0 and len(data_files["test"]) > 0:
            dataset_init = load_dataset(
                "json",
                data_files={
                    "train": data_files["train"],
                    "test": data_files["test"],
                },
            )
            print(dataset_init)
            # gather everyone if you want to have a single DatasetDict
            dataset_init = dataset_init.cast_column('tweetid', Value(dtype='int64', id=None))
            dataset_init = dataset_init.class_encode_column(stratify_column_name)
            dataset_init = dataset_init.align_labels_with_mapping(
                label2id, stratify_column_name
            )

            # train validation test split
            train_test = dataset_init["train"].train_test_split(
                test_size=0.1,
                stratify_by_column=stratify_column_name,
                seed=42,
            )

            # gather everyone if you want to have a single DatasetDict
            dataset = DatasetDict(
                {
                    "train": concatenate_datasets(
                        [train_test["train"], ds_translation["train"]]
                    )
                    if return_with_translations
                    else train_test["train"],
                    "validation": train_test["test"],
                    "test": dataset_init["test"],
                }
            )
        else:
            dataset = load_dataset(
                "json",
                data_files={
                    "train": data_files["train"] + data_files["translations"],
                    "validation": data_files["validation"],
                    "test": data_files["test"],
                },
            )
            dataset = dataset.cast_column('tweetid', Value(dtype='int64', id=None))
            dataset = dataset.class_encode_column(stratify_column_name)
            dataset = dataset.align_labels_with_mapping(label2id, stratify_column_name)

    print(dataset)
    print("end", list(set(dataset["train"]["label"])))
    print()
