# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:46:48 2021

@author: fede9
"""

import re, string, nltk
import pandas as pd
from nltk.tokenize import TweetTokenizer # tokenize tweets
from nltk.tokenize import  WordPunctTokenizer
from nltk.stem import WordNetLemmatizer # lemmatization
from nltk.stem.porter import PorterStemmer # stemming

# stopwords used
stop_words = nltk.corpus.stopwords.words("english")
   
# 1. applying regex
def preprocessing(text):
    text = text.lower() # Lowering case
    remove_url = re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', ' ', text) # Removing url
    remove_retweet = re.sub(r"@\w+", " ",remove_url) # Removing tag of users
    remove_retweet = re.sub(r"&\w+", " ",remove_retweet) # Remove &amp
    remove_retweet = re.sub(r"\b([!#\$%&\\\(\)\*\+,-\./:;<=>\?@\[\]\^_`\{|\}\"~]+)\b", " ",remove_retweet) # Must check this one
    remove_retweet = re.sub(r"([a-z])\1{3,}", r"\1",remove_retweet)
    remove_punc = remove_retweet.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    final_text = re.sub(r'\d+', ' ', remove_punc) # Remove number 
    final_text = re.sub(r'\s+', ' ', final_text) # Removing exceeding spaces
    return final_text
 
# 2. tokenization of one document
def tokenization(text_clean, tok = "tweet"):
    if tok == "tweet": # TweetTokenizer
        tt = TweetTokenizer()
        tokenized_text = tt.tokenize(text_clean)
    elif tok == "wordpunct": # WordPunctTokenizer
        wpt = WordPunctTokenizer()
        tokenized_text = wpt.tokenize(text_clean)
    return tokenized_text

# 3. remove stopwords from tokenized text
def remove_stopwords(tokenized_text):
    remove_sw = []
    for token in tokenized_text:
        stop_words.append("rt") # Added a stop words, RT for ReTweet
        if token.lower() not in stop_words:
            remove_sw.append(token)
    return remove_sw
 
# pos-tagging (1 document)
def pos_tagging(doc_token):
    return nltk.pos_tag(doc_token)

# convertion of pos tagging
def get_wordnet_pos(word_tag):
    if word_tag.startswith('J'):
        return "a"
    elif word_tag.startswith('V'):
        return "v"
    elif word_tag.startswith('R'):
        return "r"
    else:
        return "n"
    
# lemmatizer one word using pos tagging
def lemmatizer(word):
    pos = get_wordnet_pos(word[1])
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(word[0], pos = pos)

# 4. lemmatizer one document
def lemmatizer_doc(doc_token):
    lemmas = [] 
    
    pos_document = pos_tagging(doc_token) # pos tagging
    for token in pos_document:
        lemmas.append( lemmatizer(token) ) # lemmatization x word
    
    return lemmas

# 5. stemmization tokenized text
def stemmer(tokenized_text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in tokenized_text]

# process one document
def processing(text):
    text_prep = preprocessing(text)
    text_prep = tokenization(text_prep)
    text_prep = remove_stopwords(text_prep)
    text_prep = lemmatizer_doc(text_prep)
    #text_prep = stemmer(text_prep)
    text_prep = " ".join(text_prep)
    #print(text_prep)
    return text_prep