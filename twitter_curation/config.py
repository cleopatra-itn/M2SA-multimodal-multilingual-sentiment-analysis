# Update the location of input folder provided to the fetcher code
tweet_id_path = (
    "data-to-be-curated/")

# output from fetcher is stored in the following folder
dataset_location = (
    "curated-datasets/")

# API Key
# API Key Secret
bearer_token = ""

# TODO add all the files here
# "ds-name":[files....]
file_helper = {
    "twitter-15": [
        tweet_id_path + "twitter-15/Croatian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Albanian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Bosnian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Bulgarian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/English_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/German_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Hungarian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Polish_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Portuguese_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Russian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Serbian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Slovak_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Slovenian_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Spanish_Twitter_sentiment.csv",
        tweet_id_path + "twitter-15/Swedish_Twitter_sentiment.csv",
    ],
    
    "josa-corpus": [
        tweet_id_path + "josa-corpus/sa3_dev_tweetIDs.tsv",
        tweet_id_path + "josa-corpus/sa3_devBal_tweetIDs.tsv",
        tweet_id_path + "josa-corpus/sa3_test_tweetIDs.tsv",
        tweet_id_path + "josa-corpus/sa3_testBal_tweetIDs.tsv",
        tweet_id_path + "josa-corpus/sa3_train_tweetIDs.tsv",
        tweet_id_path + "josa-corpus/sa3_trainBal_tweetIDs.tsv",
    ],
    "mavis": [tweet_id_path + "MAVIS/tweets_annotations.tsv"],
    "rotulados": [tweet_id_path + "rotulados/tweets_rotulados.csv"],
    "tass2020-task10": [
        tweet_id_path + "tass2020-task1/dev_cr.tsv",
        tweet_id_path + "tass2020-task1/dev_es.tsv",
        tweet_id_path + "tass2020-task1/dev_mx.tsv",
        tweet_id_path + "tass2020-task1/dev_pe.tsv",
        tweet_id_path + "tass2020-task1/dev_uy.tsv",
        tweet_id_path + "tass2020-task1/train_cr.tsv",
        tweet_id_path + "tass2020-task1/train_es.tsv",
        tweet_id_path + "tass2020-task1/train_mx.tsv",
        tweet_id_path + "tass2020-task1/train_pe.tsv",
        tweet_id_path + "tass2020-task1/train_uy.tsv",
    ],
    "tass2019": [
        tweet_id_path+"tass2019/TASS2019_country_CR_dev.xml",
        tweet_id_path+"tass2019/TASS2019_country_CR_test.xml",
        tweet_id_path+"tass2019/TASS2019_country_CR_train.xml",
        tweet_id_path + "tass2019/TASS2019_country_ES_dev.xml",  #N##D
            tweet_id_path+"tass2019/TASS2019_country_ES_test.xml",
             tweet_id_path+"tass2019/TASS2019_country_ES_train.xml",
             tweet_id_path+"tass2019/TASS2019_country_MX_dev.xml",
             tweet_id_path+"tass2019/TASS2019_country_MX_test.xml",
             tweet_id_path+"tass2019/TASS2019_country_MX_train.xml",
             tweet_id_path+"tass2019/TASS2019_country_PE_dev.xml",
             tweet_id_path+"tass2019/TASS2019_country_PE_test.xml",
             tweet_id_path+"tass2019/TASS2019_country_PE_train.xml",
             tweet_id_path+"tass2019/TASS2019_country_UY_dev.xml",
             tweet_id_path+"tass2019/TASS2019_country_UY_test.xml",
             tweet_id_path+"tass2019/TASS2019_country_UY_train.xml",
    ],
    "tass2018-task1": [
         tweet_id_path + "tass2018-task1/intertass-CR-development-tagged.xml",
         tweet_id_path + "tass2018-task1/intertass-CR-test.xml",
         tweet_id_path + "tass2018-task1/intertass-CR-train-tagged.xml",
         tweet_id_path + "tass2018-task1/intertass-ES-development-tagged.xml",
         tweet_id_path + "tass2018-task1/intertass-ES-test.xml",
         tweet_id_path + "tass2018-task1/intertass-ES-train-tagged.xml",
         tweet_id_path + "tass2018-task1/intertass-PE-development-tagged.xml",
         tweet_id_path + "tass2018-task1/intertass-PE-test.xml",
         tweet_id_path + "tass2018-task1/intertass-PE-train-tagged.xml",
    ],
    "tass2015": [
        tweet_id_path + "tass2015/stompol-test-tagged.xml",
        tweet_id_path + "tass2015/stompol-test.xml",
        tweet_id_path + "tass2015/stompol-train-tagged.xml",
    ],
    "tass2014": [
        tweet_id_path + "tass2014/socialtv-test-tagged.xml",
        tweet_id_path + "tass2014/socialtv-test.xml",
        tweet_id_path + "tass2014/socialtv-train-tagged.xml",
     
    ],
    "tass2013": [
        tweet_id_path + "tass2013/politics-test-tagged.xml",
        
    ],
    "tass2012": [
        tweet_id_path+"tass2012/general-sentiment-1k-3l.qrel.txt", 
        tweet_id_path + "tass2012/general-sentiment-1k.qrel.txt",  
        tweet_id_path + "tass2012/general-sentiment-3l.qrel.txt",  
        tweet_id_path+"tass2012/general-sentiment.qrel.txt",
         tweet_id_path + "tass2012/general-test-tagged-3l.xml", 
        tweet_id_path + "tass2012/general-test-tagged.xml", 
         tweet_id_path + "tass2012/general-test.xml", 
         tweet_id_path + "tass2012/general-test1k-tagged-3l.xml",
         tweet_id_path + "tass2012/general-test1k-tagged.xml",
         tweet_id_path + "tass2012/general-test1k.xml",
         tweet_id_path+"tass2012/general-topics.qrel.txt",
        tweet_id_path + "tass2012/general-train-tagged-3l.xml",
         tweet_id_path + "tass2012/general-train-tagged.xml",
         tweet_id_path + "tass2012/general-users-tagged.xml",
        tweet_id_path + "tass2012/stompol-train-tagged.xml",
    ],
    "deft2015": [
        tweet_id_path + "deft2015/T1.txt", 
        tweet_id_path + "deft2015/T2.1.txt",
        tweet_id_path + "deft2015/T2.2.txt",
        tweet_id_path + "deft2015/Test-T1.txt",
        tweet_id_path + "deft2015/Test-T2.1.txt",
        tweet_id_path + "deft2015/test-T2.2.txt"
    ],
    "entityProfiling_ORM_Twitter_opinionTargets_dataset": [
        tweet_id_path +
        "entityProfiling_ORM_Twitter_opinionTargets_dataset/opinion_targets_id.tsv"
    ],
    "infocov": [
        tweet_id_path + "infocov/Senti-Cro-CoV-Twitter.csv",
    ],
    "latvian": [
        tweet_id_path + "latvian/tweet_corpus.tsv",
        tweet_id_path + "latvian/tweet_corpus_2.tsv",
    ],
    "mas_corpus": [
        tweet_id_path +
        "MAS Corpus (Corpus for Marketing Analysis in Spanish)/Total.csv"
    ],
    "train_references": [
        tweet_id_path + "Train_References-22042015/T1.txt",
        tweet_id_path + "Train_References-22042015/T2.1.txt",
        tweet_id_path + "Train_References-22042015/T3.txt",
        tweet_id_path + "Train_References-22042015/T2.2.txt",
    ],
    "xlime": [
        tweet_id_path + "xlime/german_sentiment.tsv",
        tweet_id_path + "xlime/italian_sentiment.tsv",
        tweet_id_path + "xlime/spanish_sentiment.tsv",
    ],
    "doiserbian": [tweet_id_path + "doiserbian/tweet_git.csv"],
    "sentipolc16":
    [tweet_id_path + "sentipolc16/training_set_sentipolc16.csv"],
    "Copres14": [tweet_id_path + "Copres14/dataset.tsv"],
    "GermanCovid": [
        tweet_id_path + "GermanCovidTweetCorpus/studentdata.csv",
        tweet_id_path + "GermanCovidTweetCorpus/expertdata.csv"
    ],

    "corpus-ironia-master": [
        tweet_id_path + "corpus-ironia-master/background.txt",
        tweet_id_path + "corpus-ironia-master/ironicos.txt",
        tweet_id_path + "corpus-ironia-master/noironicos.txt"
    ],
    "HUmour": [
        tweet_id_path + "HUmour/annotations_by_tweet_all.csv",
        tweet_id_path + "HUmour/annotations_by_tweet.csv",
        tweet_id_path + "HUmour/annotations.csv",
        tweet_id_path + "HUmour/tweets.csv",
    ],
    "pghumour": [
        tweet_id_path + "pghumour/corpus_public.csv",
        tweet_id_path + "pghumour/corpus.csv"
    ],
  
    "CB_COLING2014_vanzo": [
        tweet_id_path +
        "CB_COLING2014_vanzo_et_al_dataset/conv_development_set.tsv",
        tweet_id_path +
        "CB_COLING2014_vanzo_et_al_dataset/conv_testing_set.tsv",
        tweet_id_path +
        "CB_COLING2014_vanzo_et_al_dataset/conv_training_set.tsv",
        tweet_id_path +
        "CB_COLING2014_vanzo_et_al_dataset/hash_development_set.tsv",
        tweet_id_path +
        "CB_COLING2014_vanzo_et_al_dataset/hash_testing_set.tsv",
        tweet_id_path +
        "CB_COLING2014_vanzo_et_al_dataset/hash_training_set.tsv"
    ],
    "CB_IJCOL2015_ENG_castellucci": [
        tweet_id_path +
        "CB_IJCOL2015_ENG_castellucci_et_al_dataset/conv_development_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ENG_castellucci_et_al_dataset/conv_testing_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ENG_castellucci_et_al_dataset/conv_training_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ENG_castellucci_et_al_dataset/hash_development_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ENG_castellucci_et_al_dataset/hash_testing_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ENG_castellucci_et_al_dataset/hash_training_set.tsv"
    ],
    "CB_IJCOL2015_ITA_castellucci": [
        tweet_id_path +
        "CB_IJCOL2015_ITA_castellucci_et_al_dataset/conv_testing_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ITA_castellucci_et_al_dataset/conv_training_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ITA_castellucci_et_al_dataset/hash_testing_set.tsv",
        tweet_id_path +
        "CB_IJCOL2015_ITA_castellucci_et_al_dataset/hash_training_set.tsv",
    ],
    "AngryTweets": [tweet_id_path + "AngryTweets/game_tweets.csv"],
    "naija_senti_annotated_twitter_corpus": [
        tweet_id_path + "naija_senti_annotated_twitter_corpus/hausa_all.csv",
        tweet_id_path + "naija_senti_annotated_twitter_corpus/Igbo_all.csv",
        tweet_id_path + "naija_senti_annotated_twitter_corpus/pidgin_all.csv",
        tweet_id_path + "naija_senti_annotated_twitter_corpus/yo_all.csv"
    ],
    "hcr": [
        tweet_id_path + "hcr/hcr-dev.csv",
        tweet_id_path + "hcr/hcr-test.csv",
        tweet_id_path + "hcr/hcr-train.csv"
    ],
    "shamma": [tweet_id_path + "shamma/debate08_sentiment_tweets.tsv"],
    "RETWEET_data": [
        tweet_id_path + "RETWEET_data/test_gold.txt",
        tweet_id_path + "RETWEET_data/train_final_label.txt",
        tweet_id_path + "RETWEET_data/train_reply_labels_set1.txt",
        tweet_id_path + "RETWEET_data/train_reply_labels_set2.txt"
    ],
    "BounTi-Turkish-Sentiment-Analysis-main": [
        tweet_id_path + "BounTi-Turkish-Sentiment-Analysis-main/test.json",
        tweet_id_path + "BounTi-Turkish-Sentiment-Analysis-main/train.json",
        tweet_id_path +
        "BounTi-Turkish-Sentiment-Analysis-main/validation.json"
    ],
    "2017_Arabic_train_final": [
        tweet_id_path +
        "2017_Arabic_train_final/SemEval2017-task4-train.subtask-A.arabic.txt",
        tweet_id_path +
        "2017_Arabic_train_final/SemEval2017-task4-train.subtask-BD.arabic.txt",
        tweet_id_path +
        "2017_Arabic_train_final/SemEval2017-task4-train.subtask-CE.arabic.txt",
    ],
    "2017_English_final": [
        tweet_id_path + "2017_English_final/twitter-2013dev-A.txt",
        tweet_id_path + "2017_English_final/twitter-2013test-A.txt",
        tweet_id_path + "2017_English_final/twitter-2013train-A.txt",
        tweet_id_path + "2017_English_final/twitter-2014sarcasm-A.txt",
        tweet_id_path + "2017_English_final/twitter-2014test-A.txt",
        tweet_id_path + "2017_English_final/twitter-2015test-A.txt",
        tweet_id_path + "2017_English_final/twitter-2015train-A.txt",
        tweet_id_path + "2017_English_final/twitter-2016dev-A.txt",
        tweet_id_path + "2017_English_final/twitter-2016devtest-A.txt",
            
        tweet_id_path + "2017_English_final/twitter-2016test-A.txt",
        tweet_id_path + "2017_English_final/twitter-2016train-A.txt",

        
        tweet_id_path + "2017_English_final/twitter-2015test-BD.txt",
        tweet_id_path + "2017_English_final/twitter-2015train-BD.txt",
        tweet_id_path + "2017_English_final/twitter-2016dev-BD.txt",
        tweet_id_path + "2017_English_final/twitter-2016dev-CE.txt",
        tweet_id_path + "2017_English_final/twitter-2016devtest-BD.txt",
        tweet_id_path + "2017_English_final/twitter-2016test-BD.txt",
        tweet_id_path + "2017_English_final/twitter-2016test-CE.txt",
        tweet_id_path + "2017_English_final/twitter-2016devtest-CE.txt",
        tweet_id_path + "2017_English_final/twitter-2016train-BD.txt",
        tweet_id_path + "2017_English_final/twitter-2016train-CE.txt"
    
    ],
    "SemEval2015-Task10_training_subtasks_C_D": [
        tweet_id_path +
        "SemEval2015-Task10_training_subtasks_C_D/twitter-train-B-topics.txt"
    ],
    "SemEval2014-Task9-subtaskAB-test-to-download": [
        tweet_id_path +
        "SemEval2014-Task9-subtaskAB-test-to-download/SemEval2014-task9-test-A-gold-NEED-TWEET-DOWNLOAD.txt",
        tweet_id_path +
        "SemEval2014-Task9-subtaskAB-test-to-download/SemEval2014-task9-test-B-gold-NEED-TWEET-DOWNLOAD.txt"
    ],
    "SemEval2016_Task4_test_datasets-v2.0": [
        tweet_id_path +
        "SemEval2016_Task4_test_datasets-v2.0/SemEval2016-task4-test.subtask-BD.txt"
    ],
    "semeval2016-task4.traindevdevtest.v1.2": [
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_100_tweets.topic-five-point.subtask-CE.dev.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_100_tweets.topic-five-point.subtask-CE.devtest.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_100_tweets.topic-five-point.subtask-CE.train.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_XXX_tweets.topic-two-point.subtask-BD.dev.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_XXX_tweets.topic-two-point.subtask-BD.devtest.gold.txt",
        tweet_id_path +
        "semeval2016-task4.traindevdevtest.v1.2/100_topics_XXX_tweets.topic-two-point.subtask-BD.train.gold.txt"
    ],
    "TM-Senti": [
         tweet_id_path + "TM-Senti/ar-ids.tsv",
        tweet_id_path + "TM-Senti/zh-ids.tsv",
        tweet_id_path + "TM-Senti/de-ids.tsv",
        tweet_id_path + "TM-Senti/it-ids.tsv",
         tweet_id_path + "TM-Senti/fr-ids.tsv",
    ],
    "brazilian_tweet": [
        tweet_id_path + "brazilian_tweet/tweets_rotulados.csv",
    ],
    "Twi_Jap_Sentiment": [
        tweet_id_path +
        "Twitter Japanese Sentiment Analysis Dataset/tweets_open.csv",
    ],
    "bert-japan-vaccine": [
        tweet_id_path + "bert-japan-vaccine/vaccine_tweet_ids.csv",
    ],
    "Malta-Budget": [
        tweet_id_path + "malta/Malta-Budget-2018-dataset-v1.csv",
        tweet_id_path + "malta/Malta-Budget-2019-dataset-v1.csv",
        tweet_id_path + "malta/Malta-Budget-2020-dataset-v1.csv"
    ]
    # "":[
    #     tweet_id_path + "",
    # ]


}
