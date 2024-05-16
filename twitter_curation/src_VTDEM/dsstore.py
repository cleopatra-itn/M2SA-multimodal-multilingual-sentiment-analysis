import json
import config
import pandas as pd


DATASET_STORE = {
    "ar": {
        "2017_Arabic_train_final@SemEval2017-task4": {
            "train": [
                "data/final_dataset/1.2017_Arabic_train_final@SemEval2017-task4-train.subtask-A.arabic.json",
                "data/final_dataset/2.2017_Arabic_train_final@SemEval2017-task4-train.subtask-BD.arabic.json",
                "data/final_dataset/3.2017_Arabic_train_final@SemEval2017-task4-train.subtask-CE.arabic.json",
            ],
            "validation": [],
            "test": [],
        },
        "TM-Senti@ar": {
            "train": ["data/final_dataset/108.TM-Senti@ar-ids.json"],
            "validation": [],
            "test": [],
        },
    },
    "bg": {
        "twitter-15@Bulgarian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/114.twitter-15@Bulgarian_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    "bs": {
        "twitter-15@Bosnian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/115.twitter-15@Bosnian_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    "da": {
        "AngryTweets@game_tweets": {
            "train": [
                "data/final_dataset/92.AngryTweets@game_tweets.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    "de": {
        "xlime@german": {
            "train": [
                "data/final_dataset/124.xlime@german_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
        "twitter-15@German_Twitter_sentiment": {
            "train": [
                "data/final_dataset/116.twitter-15@German_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
        "TM-Senti@de": {
            "train": [
                "data/final_dataset/109.TM-Senti@de-ids.json",
            ],
            "validation": [],
            "test": [],
        },
    },
    "en": {
        "2017_English_final@twitter-2013-A": {
            "train": [
                "data/final_dataset/129.2017_English_final@twitter-2013train-A.json",
            ],
            "validation": [
                "data/final_dataset/127.2017_English_final@twitter-2013dev-A.json",
            ],
            "test": [
                "data/final_dataset/128.2017_English_final@twitter-2013test-A.json",
            ],
        },
        "2017_English_final@twitter-2014-A": {
            "train": [],
            "validation": [],
            "test": [
                "data/final_dataset/131.2017_English_final@twitter-2014test-A.json",
            ],
        },
        "2017_English_final@twitter-2015-A": {
            "train": [
                "data/final_dataset/134.2017_English_final@twitter-2015train-A.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/132.2017_English_final@twitter-2015test-A.json",
            ],
        },
        "2017_English_final@twitter-2015-BD": {
            "train": [
                "data/final_dataset/135.2017_English_final@twitter-2015train-BD.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/133.2017_English_final@twitter-2015test-BD.json",
            ],
        },
        "2017_English_final@twitter-2016-BD": {
            "train": [
                "data/final_dataset/146.2017_English_final@twitter-2016train-BD.json",
            ],
            "validation": [
                "data/final_dataset/137.2017_English_final@twitter-2016dev-BD.json",
                "data/final_dataset/140.2017_English_final@twitter-2016devtest-BD.json",
            ],
            "test": [
                "data/final_dataset/143.2017_English_final@twitter-2016test-BD.json",
            ],
        },
        "2017_English_final@twitter-2016-A": {
            "train": [
                "data/final_dataset/145.2017_English_final@twitter-2016train-A.json",
            ],
            "validation": [
                "data/final_dataset/136.2017_English_final@twitter-2016dev-A.json",
                "data/final_dataset/139.2017_English_final@twitter-2016devtest-A.json",
            ],
            "test": [
                "data/final_dataset/142.2017_English_final@twitter-2016test-A.json",
            ],
        },
        "2017_English_final@twitter-2016-CE": {
            "train": [
                "data/final_dataset/147.2017_English_final@twitter-2016train-CE.json",
            ],
            "validation": [
                "data/final_dataset/138.2017_English_final@twitter-2016dev-CE.json",
                "data/final_dataset/141.2017_English_final@twitter-2016devtest-CE.json",
            ],
            "test": [
                "data/final_dataset/144.2017_English_final@twitter-2016test-CE.json",
            ],
        },
        "CB_COLING2014_vanzo@conv": {
            "train": [
                "data/final_dataset/11.CB_COLING2014_vanzo@conv_training_set.json",
            ],
            "validation": [
                "data/final_dataset/9.CB_COLING2014_vanzo@conv_development_set.json",
            ],
            "test": [
                "data/final_dataset/10.CB_COLING2014_vanzo@conv_testing_set.json",
            ],
        },
        "CB_COLING2014_vanzo@hash": {
            "train": [
                "data/final_dataset/14.CB_COLING2014_vanzo@hash_training_set.json",
            ],
            "validation": [
                "data/final_dataset/12.CB_COLING2014_vanzo@hash_development_set.json",
            ],
            "test": [
                "data/final_dataset/13.CB_COLING2014_vanzo@hash_testing_set.json",
            ],
        },
        "CB_IJCOL2015_ENG_castellucci@conv": {
            "train": [
                "data/final_dataset/17.CB_IJCOL2015_ENG_castellucci@conv_training_set.json",
            ],
            "validation": [
                "data/final_dataset/15.CB_IJCOL2015_ENG_castellucci@conv_development_set.json",
            ],
            "test": [
                "data/final_dataset/16.CB_IJCOL2015_ENG_castellucci@conv_testing_set.json",
            ],
        },
        "CB_IJCOL2015_ENG_castellucci@hash": {
            "train": [
                "data/final_dataset/20.CB_IJCOL2015_ENG_castellucci@hash_training_set.json",
            ],
            "validation": [
                "data/final_dataset/18.CB_IJCOL2015_ENG_castellucci@hash_development_set.json",
            ],
            "test": [
                "data/final_dataset/19.CB_IJCOL2015_ENG_castellucci@hash_testing_set.json",
            ],
        },
        "RETWEET_data": {
            "train": [
                "data/final_dataset/43.RETWEET_data@train_final_label.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/42.RETWEET_data@test_gold.json",
            ],
        },
    },
    "es": {
        "xlime@spanish": {
            "train": ["data/final_dataset/126.xlime@spanish_sentiment.json"],
            "validation": [],
            "test": [],
        },
        "Copres14": {
            "train": ["data/final_dataset/25.Copres14@dataset.json"],
            "validation": [],
            "test": [],
        },
        "josa-corpus": {
            "train": [
                "data/final_dataset/34.josa-corpus@sa3_train_tweetIDs.json",
                "data/final_dataset/35.josa-corpus@sa3_trainBal_tweetIDs.json",
            ],
            "validation": [
                "data/final_dataset/30.josa-corpus@sa3_dev_tweetIDs.json",
                "data/final_dataset/31.josa-corpus@sa3_devBal_tweetIDs.json",
            ],
            "test": [
                "data/final_dataset/32.josa-corpus@sa3_test_tweetIDs.json",
                "data/final_dataset/33.josa-corpus@sa3_testBal_tweetIDs.json",
            ],
        },
        "tass2012@general": {
            "train": [
                "data/final_dataset/45.tass2012@general-sentiment-1k-qrel.json",
                "data/final_dataset/46.tass2012@general-sentiment-3l-qrel.json",
                "data/final_dataset/49.tass2012@general-train-tagged-3l.json",
                "data/final_dataset/50.tass2012@stompol-train-tagged.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/47.tass2012@general-test-tagged-3l.json",
                "data/final_dataset/48.tass2012@general-test-tagged.json",
            ],
        },
        "tass2013@politics": {
            "train": [],
            "validation": [],
            "test": [
                "data/final_dataset/51.tass2013@politics-test-tagged.json",
            ],
        },
        "tass2014@socialtv": {
            "train": [
                "data/final_dataset/54.tass2014@socialtv-train-tagged.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/52.tass2014@socialtv-test-tagged.json",
                "data/final_dataset/53.tass2014@socialtv-test.json",
            ],
        },
        "tass2015@stompol": {
            "train": [
                "data/final_dataset/57.tass2015@stompol-train-tagged.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/55.tass2015@stompol-test-tagged.json",
            ],
        },
        "tass2018-task1@intertass-CR": {
            "train": [
                "data/final_dataset/60.tass2018-task1@intertass-CR-train-tagged.json",
            ],
            "validation": [
                "data/final_dataset/58.tass2018-task1@intertass-CR-development-tagged.json",
            ],
            "test": [],
        },
        "tass2018-task1@intertass-ES": {
            "train": [
                "data/final_dataset/63.tass2018-task1@intertass-ES-train-tagged.json",
            ],
            "validation": [
                "data/final_dataset/61.tass2018-task1@intertass-ES-development-tagged.json",
            ],
            "test": [],
        },
        "tass2018-task1@intertass-PE": {
            "train": [
                "data/final_dataset/66.tass2018-task1@intertass-PE-train-tagged.json",
            ],
            "validation": [
                "data/final_dataset/64.tass2018-task1@intertass-PE-development-tagged.json",
            ],
            "test": [],
        },
        "TASS2019_country_CR": {
            "train": [
                "data/final_dataset/69.tass2019@TASS2019_country_CR_train.json",
            ],
            "validation": [
                "data/final_dataset/67.tass2019@TASS2019_country_CR_dev.json",
            ],
            "test": [
                # "data/final_dataset/68.tass2019@TASS2019_country_CR_test.json",
            ],
        },
        "TASS2019_country_ES": {
            "train": [
                "data/final_dataset/72.tass2019@TASS2019_country_ES_train.json",
            ],
            "validation": [
                "data/final_dataset/70.tass2019@TASS2019_country_ES_dev.json",
            ],
            "test": [
                # "data/final_dataset/71.tass2019@TASS2019_country_ES_test.json",
            ],
        },
        "TASS2019_country_MX": {
            "train": [
                "data/final_dataset/75.tass2019@TASS2019_country_MX_train.json",
            ],
            "validation": [
                "data/final_dataset/73.tass2019@TASS2019_country_MX_dev.json",
            ],
            "test": [
                # "data/final_dataset/74.tass2019@TASS2019_country_MX_test.json",
            ],
        },
        "TASS2019_country_PE": {
            "train": [
                "data/final_dataset/78.tass2019@TASS2019_country_PE_train.json",
            ],
            "validation": [
                "data/final_dataset/76.tass2019@TASS2019_country_PE_dev.json",
            ],
            "test": [
                # "data/final_dataset/77.tass2019@TASS2019_country_PE_test.json",
            ],
        },
        "TASS2019_country_UY": {
            "train": ["data/final_dataset/81.tass2019@TASS2019_country_UY_train.json"],
            "validation": [
                "data/final_dataset/79.tass2019@TASS2019_country_UY_dev.json",
            ],
            "test": [
                # "data/final_dataset/80.tass2019@TASS2019_country_UY_test.json",
            ],
        },
        "tass2020-task1@cr": {
            "train": [
                "data/final_dataset/87.tass2020-task1@train_cr.json",
            ],
            "validation": [
                "data/final_dataset/82.tass2020-task1@dev_cr.json",
            ],
            "test": [],
        },
        "tass2020-task1@es": {
            "train": [
                "data/final_dataset/88.tass2020-task1@train_es.json",
            ],
            "validation": [
                "data/final_dataset/83.tass2020-task1@dev_es.json",
            ],
            "test": [],
        },
        "tass2020-task1@pe": {
            "train": [
                "data/final_dataset/90.tass2020-task1@train_pe.json",
            ],
            "validation": [
                "data/final_dataset/85.tass2020-task1@dev_pe.json",
            ],
            "test": [],
        },
        "tass2020-task1@mx": {
            "train": [
                "data/final_dataset/89.tass2020-task1@train_mx.json",
            ],
            "validation": [
                "data/final_dataset/84.tass2020-task1@dev_mx.json",
            ],
            "test": [],
        },
        "tass2020-task1@uy": {
            "train": [
                "data/final_dataset/91.tass2020-task1@train_uy.json",
            ],
            "validation": [
                "data/final_dataset/86.tass2020-task1@dev_uy.json",
            ],
            "test": [],
        },
        "mavis": {
            "train": [
                "data/final_dataset/41.mavis@tweets_annotations.json",
            ],
            "validation": [],
            "test": [],
        },
        "twitter-15@Spanish_Twitter_sentiment": {
            "train": [
                "data/final_dataset/122.twitter-15@Spanish_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
    },
    "fr": {
        "deft2015@Test-T1": {
            "train": ["data/final_dataset/26.deft2015@T1.json"],
            "validation": [],
            "test": ["data/final_dataset/27.deft2015@Test-T1.json"],
        },
        "train_references@T1": {
            "train": ["data/final_dataset/112.train_references@T1.json"],
            "validation": [],
            "test": [],
        },
    },
    "hr": {
        "infocov@Senti-Cro-CoV-Twitter": {
            "train": [
                "data/final_dataset/29.infocov@Senti-Cro-CoV-Twitter.json",
            ],
            "validation": [],
            "test": [],
        },
        "twitter-15@Croatian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/115.twitter-15@Croatian_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
    },
    "hu": {
        "twitter-15@Hungarian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/117.twitter-15@Hungarian_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    "it": {
        "CB_IJCOL2015_ITA_castellucci@conv": {
            "train": [
                "data/final_dataset/22.CB_IJCOL2015_ITA_castellucci@conv_training_set.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/21.CB_IJCOL2015_ITA_castellucci@conv_testing_set.json"
            ],
        },
        "CB_IJCOL2015_ITA_castellucci@hash": {
            "train": [
                "data/final_dataset/24.CB_IJCOL2015_ITA_castellucci@hash_training_set.json",
            ],
            "validation": [],
            "test": [
                "data/final_dataset/23.CB_IJCOL2015_ITA_castellucci@hash_testing_set.json",
            ],
        },
        "xlime": {
            "train": [
                "data/final_dataset/125.xlime@italian_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
        "sentipolc16": {
            "train": [
                "data/final_dataset/44.sentipolc16@training_set_sentipolc16.json",
            ],
            "validation": [],
            "test": [],
        },
        "TM-Senti@it": {
            "train": [
                "data/final_dataset/110.TM-Senti@it-ids.json",
            ],
            "validation": [],
            "test": [],
        },
    },
    "da": {
        "AngryTweets": {
            "train": [
                "data/final_dataset/4.AngryTweets@game_tweets.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    # "lv": {
    #     "": {
    #         "train": ["data/final_dataset/37.latvian@tweet_corpus.json", ],
    #         "validation": [
    #         ],
    #         "test": [
    #         ],
    #     }
    # },
    "mt": {
        "Malta-Budget-2018": {
            "train": [
                "data/final_dataset/38.Malta-Budget@Malta-Budget-2018-dataset-v1.json",
            ],
            "validation": [],
            "test": [],
        },
        "Malta-Budget-2019": {
            "train": [
                "data/final_dataset/39.Malta-Budget@Malta-Budget-2019-dataset-v1.json"
            ],
            "validation": [],
            "test": [],
        },
        "Malta-Budget-2020": {
            "train": [
                "data/final_dataset/40.Malta-Budget@Malta-Budget-2020-dataset-v1.json"
            ],
            "validation": [],
            "test": [],
        },
    },
    "pl": {
        "twitter-15@Polish_Twitter_sentiment": {
            "train": [
                "data/final_dataset/118.twitter-15@Polish_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
    },
    "pt": {
        "twitter-15@Portuguese_Twitter_sentiment": {
            "train": [
                "data/final_dataset/119.twitter-15@Portuguese_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        },
        "brazilian_tweet@tweets_rotulados": {
            "train": ["data/final_dataset/8.brazilian_tweet@tweets_rotulados.json"],
            "validation": [],
            "test": [],
        },
    },
    "ru": {
        "twitter-15@Russian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/120.twitter-15@Russian_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    "sr": {
        "doiserbian": {
            "train": [
                "data/final_dataset/28.doiserbian@tweet_git.json",
            ],
            "validation": [],
            "test": [],
        },
        "twitter-15@Serbian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/121.twitter-15@Serbian_Twitter_sentiment.json"
            ],
            "validation": [],
            "test": [],
        },
    },
    "sv": {
        "twitter-15@Swedish_Twitter_sentiment": {
            "train": [
                "data/final_dataset/123.twitter-15@Swedish_Twitter_sentiment.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    "tr": {
        "BounTi-Turkish": {
            "train": [
                "data/final_dataset/5.Turkish-BounTi-Turkish-Sentiment-Analysis-main@train.json"
            ],
            "validation": [
                "data/final_dataset/7.BounTi-Turkish-Sentiment-Analysis-main@validation.json",
            ],
            "test": [
                "data/final_dataset/6.BounTi-Turkish-Sentiment-Analysis-main@test.json",
            ],
        }
    },
    "zh": {
        "TM-Senti@zh": {
            "train": [
                "data/final_dataset/111.TM-Senti@zh-ids.json",
            ],
            "validation": [],
            "test": [],
        }
    },
    # "lv": {
    #     "": {
    #         "train": ["data/final_dataset/37.latvian@tweet_corpus.json", ],
    #         "validation": [
    #         ],
    #         "test": [
    #         ],
    #     }
    # },
    # "sq": {
    #     "twitter-15@Albanian_Twitter_sentiment": {
    #         "train": [
    #             "data/final_dataset/113.twitter-15@Albanian_Twitter_sentiment.json",
    #         ],
    #         "validation": [
    #         ],
    #         "test": [
    #         ],
    #     }
    # },
    # this is MT version dont forget. Doing it as it cannot train-test split on its own
    "lv": {
        "latvian@tweet_corpus": {
            "train": [
                "data/final_dataset/lv_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [
                "data/final_dataset/37.latvian@tweet_corpus.json",
            ],
        }
    },
    # this is also MT version dont forget. Doing it as it cannot train-test split on its own
    "sq": {
        "twitter-15@Albanian_Twitter_sentiment": {
            "train": [
                "data/final_dataset/sq_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [
                "data/final_dataset/113.twitter-15@Albanian_Twitter_sentiment.json",
            ],
        }
    },
    "bs_mt": {
        "bs_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/bs_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "pt_mt": {
        "pt_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/pt_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "da_mt": {
        "da_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/da_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "mt_mt": {
        "mt_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/mt_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "bg_mt": {
        "bg_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/bg_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "hr_mt": {
        "hr_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/hr_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "sr_mt": {
        "sr_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/sr_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "tr_mt": {
        "tr_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/tr_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "fr_mt": {
        "fr_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/fr_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "hu_mt": {
        "hu_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/hu_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "ru_mt": {
        "ru_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/ru_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "sv_mt": {
        "sv_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/sv_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
    "zh_mt": {
        "zh_mt.43.RETWEET_data": {
            "train": [
                "data/final_dataset/zh_mt.43.RETWEET_data@train_final_label.json"
            ],
            "validation": [],
            "test": [],
        }
    },
}

# TODO return a list of train validation test file if possible else perform train test split
# hand here or at the place where its getting called.


# Assuming full langauge name is received as passed i.e., lv or lv_mt
def get_dataset_files(language, return_with_translations=False):
    file_list = {"train": [], "validation": [], "test": [], "translations": []}

    # handle translated ds first
    if return_with_translations:
        language = language.split("_")[0]
        mt_ds_collection = DATASET_STORE[language + "_mt"]  # stupidity
        for key in mt_ds_collection.keys():
            print(key)
            file_list["translations"].extend(mt_ds_collection[key]["train"])
            file_list["translations"].extend(mt_ds_collection[key]["validation"])
            file_list["translations"].extend(mt_ds_collection[key]["test"])

    if language in DATASET_STORE:
        ds_collection = DATASET_STORE[language]
        for key in ds_collection.keys():
            print(key)
            file_list["train"].extend(ds_collection[key]["train"])
            file_list["validation"].extend(ds_collection[key]["validation"])
            file_list["test"].extend(ds_collection[key]["test"])
    else:
        print("Error: language not in store", language)
    return file_list
