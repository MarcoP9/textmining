# -*- coding: utf-8 -*-

import re, string
import pandas as pd

from process_text import *

def retweet(text):
    text = preprocessing(text)
    text_tokenized = tokenization(text)
    flag = 0
    for token in text_tokenized:
        if token == "rt":
            flag = 1
    return flag

def tweet_length(text_clean):
    return len(text_clean)

def reply_tweet(text):
    if re.search("@\w+", text):
        return 1
    else:
        return 0
