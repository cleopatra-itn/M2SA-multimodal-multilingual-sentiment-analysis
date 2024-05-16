# This code creates the dataset from Corpus.csv which is downloadable from the
# internet well known dataset which is labeled manually by hand. But for the text
# of tweets you need to fetch them with their IDs.
import tweepy
import config
import logging
import csv
import urllib.request
import pandas as pd

import time, os, sys

import helpers

logger = logging.getLogger("tweepy")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="tweepy.log")
logger.addHandler(handler)

# Hydratae 
# bearer_token = ""
consumer_key = ""
consumer_key_secret = ""
access_token = "-db1fgQDEr8n0votMxHr3LaYfuLKu6qr"
access_token_secret = ""


bearer_token = config.bearer_token

client = tweepy.Client(bearer_token= config.bearer_token, consumer_key=consumer_key,
                       consumer_secret=consumer_key_secret,
                       access_token=access_token,
                       access_token_secret=access_token_secret,
                       wait_on_rate_limit=True)


def createTrainingSet(corpusFile, targetResultFile):
    image_location = targetResultFile.split(".")[0]
    counter = 0
    corpus = []

    with open(corpusFile, "r") as csvfile:
        lineReader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(lineReader)
        for index, row in enumerate(lineReader):
            # if row[0] in done_ids: continue
            corpus.append(
                {"AnnotatorID": row[2], "HandLabel": row[1], "TweetID": row[0]}
            )
            if index % 1000 == 0:
                print("index", index)

    sleepTime = 2
    # trainingDataSet = []
    with open(targetResultFile, "w") as csvfile, \
        open("empty_tweets.txt", "a") as emptyfile:
        # TODO update the column headers when adding something new
        csvfile.write("\t".join(["TweetID","text","HandLabel","created_on","file_names","AnnotatorID\n"]))
        for index, tweet in enumerate(corpus):
            try:
                print(tweet["TweetID"])
                tweetFetched = client.get_tweet(
                    tweet["TweetID"],  
                    expansions=["attachments.media_keys", "author_id", "geo.place_id"],
                    media_fields=[
                        "duration_ms",
                        "height",
                        "media_key",
                        "preview_image_url",
                        "public_metrics",
                        "type",
                        "url",
                        "width",
                        "alt_text",
                    ],
                    tweet_fields=["created_at", "geo"],
                )
                # handle text
                print("Tweet fetched" + tweetFetched.data.text)
                tweet["text"] = tweetFetched.data.text.replace("\n", " ")
                tweet["created_on"] = str(tweetFetched.data.created_at)
                
                # trainingDataSet.append(tweet)
                
                media_file_names = []
                if tweetFetched.includes and "media" in tweetFetched.includes:
                    for media in tweetFetched.includes["media"]:
                        if media["type"] == "video":
                            urllib.request.urlretrieve(
                                media["preview_image_url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                        elif media["type"] == "photo":
                            urllib.request.urlretrieve(
                                media["url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                
                # save file if exists
                # TODO update the column headers when adding something new
                csvfile.write(
                    "\t".join(
                        [
                            tweet["TweetID"],
                            tweet["text"],
                            tweet["HandLabel"],
                            tweet["created_on"],
                            ",".join(media_file_names), tweet["AnnotatorID"]
                        ]
                    )
                )
                csvfile.write("\n")
            except Exception as e:
                print(e)
                print(tweet["TweetID"], "Inside the exception - no:2")
                emptyfile.write(tweet["TweetID"] + "\n")
                continue
def createJosaSingleTrainingSet(corpusFile, targetResultFile):
    image_location = targetResultFile.split(".")[0]
    counter = 0
    corpus = []

    with open(corpusFile, "r") as csvfile:
        lineReader = csv.reader(csvfile, delimiter="\t", quotechar='"')
        next(lineReader)
        for index, row in enumerate(lineReader):
            # if row[0] in done_ids: continue
            corpus.append(
                {"HandLabel": row[1], "TweetID": row[0]}
            )
            if index % 1000 == 0:
                print("index", index)

    sleepTime = 2
    # trainingDataSet = []
    with open(targetResultFile, "w") as csvfile, \
        open("empty_tweets.txt", "a") as emptyfile:
        # TODO update the column headers when adding something new
        csvfile.write("\t".join(["TweetID","text","HandLabel","created_on","file_names\n"]))
        for index, tweet in enumerate(corpus):
            try:
                print(tweet["TweetID"])
                tweetFetched = client.get_tweet(
                    tweet["TweetID"],  # ,#1530155201524813825, 1530112485658877952
                    expansions=["attachments.media_keys", "author_id", "geo.place_id"],
                    media_fields=[
                        "duration_ms",
                        "height",
                        "media_key",
                        "preview_image_url",
                        "public_metrics",
                        "type",
                        "url",
                        "width",
                        "alt_text",
                    ],
                    tweet_fields=["created_at", "geo"],
                )
                # handle text
                print("Tweet fetched" + tweetFetched.data.text)
                tweet["text"] = tweetFetched.data.text.replace("\n", " ")
                tweet["created_on"] = str(tweetFetched.data.created_at)
                
                # trainingDataSet.append(tweet)
                
                media_file_names = []
                if tweetFetched.includes and "media" in tweetFetched.includes:
                    for media in tweetFetched.includes["media"]:
                        if media["type"] == "video":
                            urllib.request.urlretrieve(
                                media["preview_image_url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                        elif media["type"] == "photo":
                            urllib.request.urlretrieve(
                                media["url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                
                # save file if exists
                # TODO update the column headers when adding something new
                csvfile.write(
                    "\t".join(
                        [
                            tweet["TweetID"],
                            tweet["text"],
                            tweet["HandLabel"],
                            tweet["created_on"],
                            ",".join(media_file_names)
                        ]
                    )
                )
                csvfile.write("\n")
            except Exception as e:
                print(e)
                print(tweet["TweetID"], "Inside the exception - no:2")
                emptyfile.write(tweet["TweetID"] + "\n")
                continue
def createMavisTrainingSet(corpusFile, targetResultFile):
    image_location = targetResultFile.split(".")[0]
    counter = 0
    corpus = []

    with open(corpusFile, "r") as csvfile:
        lineReader = csv.reader(csvfile, delimiter="\t", quotechar='"')
        next(lineReader)
        for index, row in enumerate(lineReader):
            # if row[0] in done_ids: continue
            corpus.append(
                {
                   "TweetID": row[0],
                    
                }
            )
            if index % 1000 == 0:
                print("index", index)

    with open(targetResultFile, "w") as csvfile, \
        open("empty_tweets.txt", "a") as emptyfile:
        # TODO update the column headers when adding something new
        csvfile.write("\t".join(["TweetID","text","created_on","file_names\n"]))
        for index, tweet in enumerate(corpus):
            try:
                print(tweet["TweetID"])
                tweetFetched = client.get_tweet(
                    tweet["TweetID"],  # ,#1530155201524813825, 1530112485658877952
                    expansions=["attachments.media_keys", "author_id", "geo.place_id"],
                    media_fields=[
                        "duration_ms",
                        "height",
                        "media_key",
                        "preview_image_url",
                        "public_metrics",
                        "type",
                        "url",
                        "width",
                        "alt_text",
                    ],
                    tweet_fields=["created_at", "geo"],
                )
                # handle text
                print("Tweet fetched" + tweetFetched.data.text)
                tweet["text"] = tweetFetched.data.text.replace("\n", " ")
                tweet["created_on"] = str(tweetFetched.data.created_at)
                
                # trainingDataSet.append(tweet)
                
                media_file_names = []
                if tweetFetched.includes and "media" in tweetFetched.includes:
                    for media in tweetFetched.includes["media"]:
                        if media["type"] == "video":
                            urllib.request.urlretrieve(
                                media["preview_image_url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                        elif media["type"] == "photo":
                            urllib.request.urlretrieve(
                                media["url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                
                # save file if exists
                # TODO update the column headers when adding something new
                csvfile.write(
                    "\t".join(
                        [
                            tweet["TweetID"],
                            tweet["text"],
                   
                            tweet["created_on"],
                            ",".join(media_file_names)
                        ]
                    )
                )
                csvfile.write("\n")
            except Exception as e:
                print(e)
                print(tweet["TweetID"], "Inside the exception - no:2")
                emptyfile.write(tweet["TweetID"] + "\n")
                continue
def createRotuladosTrainingSet(corpusFile, targetResultFile):
    image_location = targetResultFile.split(".")[0]
    counter = 0
    corpus = []

    with open(corpusFile, "r") as csvfile:
        lineReader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(lineReader)
        for index, row in enumerate(lineReader):
            # if row[0] in done_ids: continue
            corpus.append(
                {
                   "TweetID": row[2],
                    
                }
            )
            if index % 1000 == 0:
                print("index", index)

    with open(targetResultFile, "w") as csvfile, \
        open("empty_tweets.txt", "a") as emptyfile:
        # TODO update the column headers when adding something new
        csvfile.write("\t".join(["TweetID","text","created_on","file_names\n"]))
        for index, tweet in enumerate(corpus):
            try:
                print(tweet["TweetID"])
                tweetFetched = client.get_tweet(
                    tweet["TweetID"],  # ,#1530155201524813825, 1530112485658877952
                    expansions=["attachments.media_keys", "author_id", "geo.place_id"],
                    media_fields=[
                        "duration_ms",
                        "height",
                        "media_key",
                        "preview_image_url",
                        "public_metrics",
                        "type",
                        "url",
                        "width",
                        "alt_text",
                    ],
                    tweet_fields=["created_at", "geo"],
                )
                # handle text
                print("Tweet fetched" + tweetFetched.data.text)
                tweet["text"] = tweetFetched.data.text.replace("\n", " ")
                tweet["created_on"] = str(tweetFetched.data.created_at)
                
                # trainingDataSet.append(tweet)
                
                media_file_names = []
                if tweetFetched.includes and "media" in tweetFetched.includes:
                    for media in tweetFetched.includes["media"]:
                        if media["type"] == "video":
                            urllib.request.urlretrieve(
                                media["preview_image_url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                        elif media["type"] == "photo":
                            urllib.request.urlretrieve(
                                media["url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                
                # save file if exists
                # TODO update the column headers when adding something new
                csvfile.write(
                    "\t".join(
                        [
                            tweet["TweetID"],
                            tweet["text"],
                   
                            tweet["created_on"],
                            ",".join(media_file_names)
                        ]
                    )
                )
                csvfile.write("\n")
            except Exception as e:
                print(e)
                print(tweet["TweetID"], "Inside the exception - no:2")
                emptyfile.write(tweet["TweetID"] + "\n")
                continue

def createTASS2020T1TrainingSet(corpusFile, targetResultFile):
    image_location = targetResultFile.split(".")[0]
    counter = 0
    corpus = []

    with open(corpusFile, "r") as csvfile:
        lineReader = csv.reader(csvfile, delimiter="\t", quotechar='"')
        # next(lineReader)
        for index, row in enumerate(lineReader):
            # if row[0] in done_ids: continue
            corpus.append(
                {
                   "TweetID": row[0],
                    
                }
            )
            if index % 1000 == 0:
                print("index", index)

    with open(targetResultFile, "w") as csvfile, \
        open("empty_tweets.txt", "a") as emptyfile:
        # TODO update the column headers when adding something new
        csvfile.write("\t".join(["TweetID","text","created_on","file_names\n"]))
        for index, tweet in enumerate(corpus):
            try:
                print(tweet["TweetID"])
                tweetFetched = client.get_tweet(
                    tweet["TweetID"],  # ,#1530155201524813825, 1530112485658877952
                    expansions=["attachments.media_keys", "author_id", "geo.place_id"],
                    media_fields=[
                        "duration_ms",
                        "height",
                        "media_key",
                        "preview_image_url",
                        "public_metrics",
                        "type",
                        "url",
                        "width",
                        "alt_text",
                    ],
                    tweet_fields=["created_at", "geo"],
                )
                # handle text
                print("Tweet fetched" + tweetFetched.data.text)
                tweet["text"] = tweetFetched.data.text.replace("\n", " ")
                tweet["created_on"] = str(tweetFetched.data.created_at)
                
                # trainingDataSet.append(tweet)
                
                media_file_names = []
                if tweetFetched.includes and "media" in tweetFetched.includes:
                    for media in tweetFetched.includes["media"]:
                        if media["type"] == "video":
                            urllib.request.urlretrieve(
                                media["preview_image_url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                        elif media["type"] == "photo":
                            urllib.request.urlretrieve(
                                media["url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                
                # save file if exists
                # TODO update the column headers when adding something new
                csvfile.write(
                    "\t".join(
                        [
                            tweet["TweetID"],
                            tweet["text"],
                            tweet["created_on"],
                            ",".join(media_file_names)
                        ]
                    )
                )
                csvfile.write("\n")
            except Exception as e:
                print(e)
                print(tweet["TweetID"], "Inside the exception - no:2")
                emptyfile.write(tweet["TweetID"] + "\n")
                continue
def createTASS2019TrainingSet(corpusFile, targetResultFile):
    print("fname",corpusFile)
    image_location, ext = targetResultFile.rsplit(".",1)

    corpus = []
    col_name = ""
    df = None
    if ext in ["xml"]:
        df = pd.read_xml(corpusFile)
    elif ext in ["tsv"]:
        df = pd.read_csv(corpusFile,sep="\t")
        df.columns = ["date","tweetid","label","emoticons"] #"tweetid","label","labelsource"

    elif ext in ["csv"]:
        df = pd.read_csv(corpusFile)
        # df.columns = ["tweetid","label"]
        # df.rename(columns={ df.columns[0]: "tweetid"}, inplace=True)
    elif ext in ["txt"]:
        df = pd.read_csv(corpusFile,sep="\t") #sep=":"
        # df.columns = ["tweetid","label"]
        df.rename(columns={ df.columns[0]: "tweetid"}, inplace=True)
    if ext in ["json"]:
        df = pd.read_json(corpusFile)
    #check if the df has header if not then set the first column to tweetid
    # if not isinstance(df.columns[0], str):
    #     df.rename(columns={ df.columns[0]: "tweetid"}, inplace=True)

    if "tweetid" in df.columns:
        col_name = "tweetid"
    elif "tweet.id" in df.columns:
        col_name = "tweet.id"
    elif "tweet_id" in df.columns:
        col_name = "tweet_id"
    elif "tweet id" in df.columns:
        col_name = "tweet id"
    elif "id_tweet" in df.columns:
        col_name = "id_tweet"
    elif "Twitter ID" in df.columns:
        col_name = "Twitter ID"
    elif "id" in df.columns:
        col_name = "id"

    df.drop_duplicates(subset=[col_name],inplace=True)
    df.dropna(subset=[col_name],inplace=True)
    
    if "Source ID" in df.columns:
        df = df[df["Source ID"].isin(["SNS-1-2018","SNS-1-2019","SNS-1-2020"])]
        df[col_name] = df[col_name].astype(int).astype(str)

    for _, row in df.iterrows():
        print(row[col_name])
        corpus.append(
                {
                   "TweetID": str(row[col_name]),
                }
        )

    with open(targetResultFile, "w") as csvfile, \
        open("empty_tweets.txt", "a") as emptyfile:
        # TODO update the column headers when adding something new
        csvfile.write("\t".join(["TweetID","text","created_on","file_names\n"]))
        for index, tweet in enumerate(corpus):
            try:
                print(tweet["TweetID"])
                tweetFetched = client.get_tweet(
                    tweet["TweetID"],  # ,#1530155201524813825, 1530112485658877952
                    expansions=["attachments.media_keys", "author_id", "geo.place_id"],
                    media_fields=[
                        "duration_ms",
                        "height",
                        "media_key",
                        "preview_image_url",
                        "public_metrics",
                        "type",
                        "url",
                        "width",
                        "alt_text",
                    ],
                    tweet_fields=["created_at", "geo"],
                )
                # handle text
                print("Tweet fetched" + tweetFetched.data.text)
                tweet["text"] = tweetFetched.data.text.replace("\n", " ")
                tweet["created_on"] = str(tweetFetched.data.created_at)
                
                # trainingDataSet.append(tweet)
                
                media_file_names = []
                if tweetFetched.includes and "media" in tweetFetched.includes:
                    for media in tweetFetched.includes["media"]:
                        if media["type"] == "video":
                            urllib.request.urlretrieve(
                                media["preview_image_url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                        elif media["type"] == "photo":
                            urllib.request.urlretrieve(
                                media["url"],
                                image_location + "/" + media["media_key"] + ".jpg",
                            )
                            media_file_names.append(media["media_key"] + ".jpg")
                
                # save file if exists
                # TODO update the column headers when adding something new
                csvfile.write(
                    "\t".join(
                        [
                            tweet["TweetID"],
                            tweet["text"],
                            tweet["created_on"],
                            ",".join(media_file_names)
                        ]
                    )
                )
                csvfile.write("\n")
                csvfile.flush()
            except Exception as e:
                print(e)
                print(tweet["TweetID"], "Inside the exception - no:2")
                emptyfile.write(tweet["TweetID"] + "\n")
                continue
if __name__ == "__main__":
    # Code starts here
    # loop through datasets for name
    for dataset_name in [
                         "AngryTweets",  
                         "hcr",
                         "2017_Arabic_train_final", 
                         "shamma", 
                         "RETWEET_data", 
                         "brazilian_tweet", 
                         "Malta-Budget" 
                         "CB_IJCOL2015_ITA_castellucci", 
                         "CB_COLING2014_vanzo",
                         "CB_IJCOL2015_ENG_castellucci",
                         "BounTi-Turkish-Sentiment-Analysis-main",
                            "2017_English_final",
                         "TM-Senti",
                         "SemEval2015-Task10_training_subtasks_C_D",
                         "SemEval2014-Task9-subtaskAB-test-to-download",
                         "SemEval2016_Task4_test_datasets-v2.0",
                         "semeval2016-task4.traindevdevtest.v1.2",
                         "bert-japan-vaccine",
                        ]:  
      

        # for every dataset loop through files
        for twitter_id_file in config.file_helper[dataset_name]:
            # for every file, generate its saving location
            path_helper = helpers.PathHelper(dataset_name=dataset_name,
                                             file_path=twitter_id_file)
            result_file_location = path_helper.get_output_file_path()
            
            # specific datasets with specific code
            
            # if dataset_name == "mavis":
            #     createMavisTrainingSet(twitter_id_file, result_file_location)
            # else:   
            #     createJosaSingleTrainingSet(twitter_id_file, result_file_location)
            # createTrainingSet(twitter_id_file, result_file_location)
            # createRotuladosTrainingSet(twitter_id_file, result_file_location)
            # createTASS2020T1TrainingSet(twitter_id_file, result_file_location)
            
            createTASS2019TrainingSet(twitter_id_file, result_file_location)