#!/usr/bin/env bash

#
# CREDIT: Delip Rao: Natural Language Processing with Pytorch
#

# For each file, add a download.py line
# Any additional processing on the downloaded file

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR=~/work/transfer-nlp-data

# Yelp Reviews dataset
mkdir -p $DATA_DIR/yelp
if [ ! -f $DATA_DIR/yelp/raw_train.csv ]; then
    python $HERE/download.py 1xeUnqkhuzGGzZKThzPeXe2Vf6Uu_g_xM $DATA_DIR/yelp/raw_train.csv # 12536
fi
if [ ! -f $DATA_DIR/yelp/raw_test.csv ]; then
    python $HERE/download.py 1G42LXv72DrhK4QKJoFhabVL4IU6v2ZvB $DATA_DIR/yelp/raw_test.csv # 4
fi
if [ ! -f $DATA_DIR/yelp/reviews_with_splits_lite.csv ]; then
    python $HERE/download.py 1Lmv4rsJiCWVs1nzs4ywA9YI-ADsTf6WB $DATA_DIR/yelp/reviews_with_splits_lite.csv # 1217
fi

# Surnames Dataset
mkdir -p $DATA_DIR/surnames
if [ ! -f $DATA_DIR/surnames/surnames.csv ]; then
    python $HERE/download.py 1MBiOU5UCaGpJw2keXAqOLL8PCJg_uZaU $DATA_DIR/surnames/surnames.csv # 6
fi
if [ ! -f $DATA_DIR/surnames/surnames_with_splits.csv ]; then
    python $HERE/download.py 1T1la2tYO1O7XkMRawG8VcFcvtjbxDqU- $DATA_DIR/surnames/surnames_with_splits.csv # 8
fi

# Books Dataset
mkdir -p $DATA_DIR/books
if [ ! -f $DATA_DIR/books/frankenstein.txt ]; then
    python $HERE/download.py 1XvNPAjooMyt6vdxknU9VO_ySAFR6LpAP $DATA_DIR/books/frankenstein.txt # 14
fi
if [ ! -f $DATA_DIR/books/frankenstein_with_splits.csv ]; then
    python $HERE/download.py 1dRi4LQSFZHy40l7ZE85fSDqb3URqh1Om $DATA_DIR/books/frankenstein_with_splits.csv # 109

fi

# AG News Dataset
mkdir -p $DATA_DIR/ag_news
if [ ! -f $DATA_DIR/ag_news/news.csv ]; then
    python $HERE/download.py 1hjAZJJVyez-tjaUSwQyMBMVbW68Kgyzn $DATA_DIR/ag_news/news.csv # 188
fi
if [ ! -f $DATA_DIR/ag_news/news_with_splits.csv ]; then
    python $HERE/download.py 1Z4fOgvrNhcn6pYlOxrEuxrPNxT-bLh7T $DATA_DIR/ag_news/news_with_splits.csv # 208
fi

mkdir -p $DATA_DIR/nmt
if [ ! -f $DATA_DIR/nmt/eng-fra.txt ]; then
    python $HERE/download.py 1o2ac0EliUod63sYUdpow_Dh-OqS3hF5Z $DATA_DIR/nmt/eng-fra.txt # 292
fi
if [ ! -f $DATA_DIR/nmt/simplest_eng_fra.csv ]; then
    python $HERE/download.py 1jLx6dZllBQ3LXZkCjZ4VciMQkZUInU10 $DATA_DIR/nmt/simplest_eng_fra.csv # 30
fi