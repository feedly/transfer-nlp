"""
This file aims at using the python wrapper around the Feedly API to build a dataset of articles given a feed name
Note: You need a Feedly account and a token to use this script.
Visit the Feedly API page to generate a token: https://developer.feedly.com/
"""

from pathlib import Path
from random import shuffle
from typing import List

import pandas as pd
import urllib3
from bs4 import BeautifulSoup
from feedly.data import StreamOptions, Entry
from feedly.session import FeedlySession

urllib3.disable_warnings()


def get_text(entry: Entry) -> BeautifulSoup:

  full_content = entry.json["fullContent"] if "fullContent" in entry.json else ""
  content = entry.json["content"]["content"] if "content" in entry.json else ""
  summary = entry.json["summary"]["content"] if "summary" in entry.json else ""
  title = entry.json["title"]
  best=max(full_content, content, summary, title, key=len)

  return BeautifulSoup(best.replace("\n", ""), 'html.parser').text


def build_dataframe(entries: List[Entry]) -> pd.DataFrame:

    eid = [e.json.get("id") for e in entries]  # id of the entry
    title = [e.json["title"] for e in entries]  # title of the entry
    content = [get_text(e) for e in entries]  # text content of the entry
    data = {
        'eid': eid,
        'title': title,
        'content': content}

    return pd.DataFrame(data=data)


class FeedlyDownloader:

    def __init__(self, token: str):

        self.token = token
        self.df: pd.DataFrame = None

    def get_entries(self, category: str, max_count: int) -> List[Entry]:

        with FeedlySession(auth=self.token) as sess:
            feeds = sess.user.get_enterprise_categories()
            keep = None
            for feed in feeds:
                if feeds[feed].json['label'] == category:
                    keep = feeds[feed]
            category_id = keep.stream_id.content_id
            entries = sess.user.get_enterprise_category(category_id).stream_contents(options=StreamOptions(max_count=max_count))
            entries = list(entries)

        return entries

    def build_dataset(self, category: str, max_count: int, save_path: Path):

        entries = self.get_entries(category=category, max_count=max_count)
        shuffle(entries)
        self.df = build_dataframe(entries)
        limits = [int(0.8*len(entries)), int(0.1*len(entries))]
        split = ['train'] * limits[0] + ['val'] * limits[1]
        split.extend(['test'] * (len(entries) - len(split)))
        self.df['split'] = split
        self.df['nationality'] = ['en'] * len(entries)

        self.df.to_csv(path_or_buf=save_path)


if __name__ == "__main__":

    token = "YourToken"
    downloader = FeedlyDownloader(token=token)
    save_path = Path.home() / 'work/experiments/nlp/data'
    category = 'YourFeed'
    max_count = 10000
    save_path = save_path / 'feedly_data10000.csv'
    downloader.build_dataset(category=category, max_count=max_count, save_path=save_path)

