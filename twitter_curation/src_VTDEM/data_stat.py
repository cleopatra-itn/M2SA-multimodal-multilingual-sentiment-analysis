import dsstore as store
import pandas as pd
from datasets import (
    load_dataset,
    DatasetDict,
    disable_caching,
    concatenate_datasets,
    Value,
)

disable_caching()

import re

def compute_average_tweet_length(tweets):
  """Computes the average number of words in a list of tweets.

  Args:
    tweets: A list of strings, where each string represents a tweet.

  Returns:
    A float representing the average number of words in the tweets.
  """

  # Split each tweet into words.
  words = []
  for tweet in tweets:
    tweet_words = re.split(r'\s+', tweet)
    words.extend(tweet_words)

  # Calculate the average number of words.
  average_tweet_length = sum([len(word) for word in words]) / len(words)

  return average_tweet_length

pd_collection = []
 
for key in store.DATASET_STORE.keys():
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
    files = []
    [files.extend(i) for i in data_files.values()]

    
    texts = []
   
    for data_file in files:
        ds_translation = []
        
        langs = []
        ds_name = []
        positives = []
        negatives = []
        neutrals = []
        print()
        # print("file:", data_file.replace("data/final_dataset/", ""))
        df = pd.read_json(data_file, lines=True)
        text = df.normalized_text.values.tolist()
        texts.extend(text)
        
        
    print("count_average_tokens(text)",key, compute_average_tweet_length(texts))

