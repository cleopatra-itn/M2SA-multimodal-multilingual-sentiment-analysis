import config
import os
from pathlib import Path
import urllib
import sys

class PathHelper:
    def __init__(self, dataset_name, file_path) -> None:
        self._file_path = file_path
        self._folder_path = file_path
        self._file_name = Path(file_path).name  #file_path.split("_")[0]
        self._file_parent_folder_name = Path(file_path).stem
        self._dataset_name = dataset_name

    def get_output_file_path(self):
        # Check whether the specified path exists or not
        # This is for images: output_folder location + "twitter15" + "german"
        path = os.path.join(config.dataset_location, self._dataset_name,
                            self._file_parent_folder_name)
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
        #
        # this is for csv # output_folder location + "twitter15" + "german.csv"
        return os.path.join(config.dataset_location, self._dataset_name,
                            self._file_name)


class TweetFetcher:
    def __init__(self, client):
        self.client = client

    def fetch(self, tweet_ids, emptyfile):
        '''
        corpus : List[{"TweetID":xxxx}]
        '''
       
        try:
            print(tweet_ids)
            tweetsFetched = self.client.get_tweets(
             ",".join([ str(i) for i in set(tweet_ids)]),
            expansions=[
                "attachments.media_keys", "author_id", "geo.place_id"
            ],
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
            
            print(["includes" in i for i in tweetsFetched.data]
)
            return  tweetsFetched
        except Exception as e:
            print(e)
            # print(tweet["TweetID"], "Inside the exception - no:2")
            # emptyfile.write(tweet["TweetID"] + "\n")
            
