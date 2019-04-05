"""
This file aims at using the python wrapper around the Feedly API to build a dataset of articles given a feed name
Note: You need a Feedly account and a token to use this script.
Visit the Feedly API page to generate a token: https://developer.feedly.com/
"""

import os
from pathlib import Path
from random import shuffle
from typing import List

import numpy as np
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

    def get_category_entries(self, category: str, max_count: int, account: str='enterprise') -> List[Entry]:

        with FeedlySession(auth=self.token) as sess:
            if account == 'enterprise':
                feeds = sess.user.get_enterprise_categories()
            elif account == 'personal':
                feeds = sess.user.get_categories()
            else:
                raise ValueError("Only enterprise and personal account are taken into account")
            keep = None
            for feed in feeds:
                if feeds[feed].json['label'] == category:
                    keep = feeds[feed]
            print(keep)
            category_id = keep.stream_id.content_id
            entries = []
            if account == 'enterprise':
                entries = sess.user.get_enterprise_category(category_id).stream_contents(options=StreamOptions(max_count=max_count))
            elif account == 'personal':
                entries = sess.user.get_category(category_id).stream_contents(options=StreamOptions(max_count=max_count))
            else:
                raise ValueError("Only enterprise and personal account are taken into account")
            entries = list(entries)

        return entries


    def get_board_entries(self, board: str, max_count: int, account: str='enterprise') -> List[Entry]:

        with FeedlySession(auth=self.token) as sess:
            if account == 'enterprise':
                feeds = sess.user.get_enterprise_tags()
            elif account == 'personal':
                feeds = sess.user.get_tags()
            else:
                raise ValueError("Only enterprise and personal account are taken into account")
            keep = None
            for feed in feeds:
                if feeds[feed].json['label'] == board:
                    keep = feeds[feed]
            print(keep)
            category_id = keep.stream_id.content_id
            entries = []
            if account == 'enterprise':
                entries = sess.user.get_enterprise_tag(category_id).stream_contents(options=StreamOptions(max_count=max_count))
            elif account == 'personal':
                entries = sess.user.get_tag(category_id).stream_contents(options=StreamOptions(max_count=max_count))
            else:
                raise ValueError("Only enterprise and personal account are taken into account")
            entries = list(entries)

        return entries


    def build_dataset(self, category: str, max_count: int, save_path: Path):

        entries = self.get_category_entries(category=category, max_count=max_count)
        shuffle(entries)
        self.df = build_dataframe(entries)
        limits = [int(0.8*len(entries)), int(0.1*len(entries))]
        split = ['train'] * limits[0] + ['val'] * limits[1]
        split.extend(['test'] * (len(entries) - len(split)))
        self.df['split'] = split
        self.df['nationality'] = ['en'] * len(entries)

        self.df.to_csv(path_or_buf=save_path)

    def build_multi_class_dataset(self, categories: List[str], max_count: int, save_path: Path):

        entries = {category: self.get_category_entries(category=category, max_count=max_count, account='personal') for category in categories}
        [shuffle(entries[category]) for category in entries]
        entries = {category: build_dataframe(entries[category]) for category in entries}
        for category in entries:
            entries[category]['class'] = category
        df = pd.concat([entries[category] for category in entries])
        np.random.shuffle(df.values)
        limits = [int(0.8*len(df)), int(0.1*len(df))]
        split = ['train'] * limits[0] + ['val'] * limits[1]
        split.extend(['test'] * (len(df) - len(split)))
        df['split'] = split
        self.df = df
        self.df.to_csv(path_or_buf=save_path)

    def build_like_board_dataset(self, category: str, boards: List[str], max_count: int, save_path: Path):

        category_entries = self.get_category_entries(category=category, max_count=max_count)
        shuffle(category_entries)
        df_category = build_dataframe(entries=category_entries)
        df_category['category'] = ['category'] * len(category_entries)
        boards_entries = []
        for board in boards:
            boards_entries.extend(self.get_board_entries(board=board, max_count=max_count))
        shuffle(boards_entries)
        df_boards = build_dataframe(entries=boards_entries)
        df_boards['category'] = ['board'] * len(df_boards)

        for df in [df_category, df_boards]:
            limits = [int(0.8 * len(df)), int(0.1 * len(df))]
            split = ['train'] * limits[0] + ['val'] * limits[1]
            split.extend(['test'] * (len(df) - len(split)))
            df['split'] = split

        df = pd.concat([df_category, df_boards])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(path_or_buf=save_path)


if __name__ == "__main__":

    token ="YourToken"
    downloader = FeedlyDownloader(token=token)

    # # One category
    # save_path = Path.home() / 'work/experiments/nlp/data'
    # category = 'YourFeed'
    # max_count = 10000
    # save_path = save_path / 'feedly_data10000.csv'
    # downloader.build_dataset(category=category, max_count=max_count, save_path=save_path)
    #
    # #Multilingual
    # token = "YourToken"
    # downloader = FeedlyDownloader(token=token)
    # categories = ["Category1", "Category2"]
    # max_count = 1000
    # save_path = Path.home() / 'work/experiments/nlp/data/feedly_multilingual.csv'
    # downloader.build_multi_class_dataset(categories=categories, max_count=max_count, save_path=save_path)

    # # One category, One board
    # category = "YourCategory"
    # boards = ['YourFirstBoard', 'YourSecondBoard']
    # max_count = 100
    # save_path = save_path / 'feedly_data_lb.csv'
    # downloader.build_dataset(category=category, boards=boards, max_count=max_count, save_path=save_path)



