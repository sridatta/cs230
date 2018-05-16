import pandas as pd
from twarc import Twarc
import glob
import os.path

# Twitter auth for downloading tweets
CONSUMER_KEY =  os.environ.get("TWITTER_CONSUMER_KEY")
CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")
ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

# Concat and read all the CSVs
dir1 = "data/twitter-framing-master/congressional_tweets_dataset_2017/unlabeled/"
dir2 = "data/twitter-framing-master/congressional_tweets_dataset_2017/labeled/"
csv_files = glob.glob(os.path.join(dir1, "*.csv")) + glob.glob(os.path.join(dir2, "*.csv"))
HEADERS = ["tweet_id", "issue1", "issue2", "frame1", "frame2", "frame3", "party", "ts"]
all_df = pd.concat((pd.read_csv(f, names=HEADERS, header=None) for f in csv_files), ignore_index=True)

t = Twarc(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
tweet_texts = {}
for tweet in t.hydrate(all_df["tweet_id"]):
    tweet_texts[tweet["id"]] = tweet["full_text"]

text_df = pd.DataFrame(tweet_texts, index=[0]).transpose().rename(columns={"index": "tweet_id", 0: "text"})
all_df = all_df.set_index("tweet_id")
joined = all_df.join(text_df)
joined.to_pickle("data/tweets.pkl")
