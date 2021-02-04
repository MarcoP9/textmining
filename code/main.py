import nltk
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument # doc2vec
import pandas as pd

#setting working directory
os.chdir("C:/Users/fede9/Documents/GitHub/textmining/code")
from process_text import processing

if __name__ == "__main__":
   # downloads
   nltk.download("stopwords")
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   
   print("***** TEXT MINING Project *****")
   
   ### process_text
   if 'processed_data.csv' not in os.listdir('../data'):
      # load dataset
      df = pd.read_csv("../data/labeled_data.csv", sep = ",").drop("Unnamed: 0", axis=1)
      # apply process_text
      df["tweet_clean"] = df["tweet"].apply(lambda x : processing(x))
      df.to_csv('../data/processed_data.csv', index = False)
   else:
      df = pd.read_csv("../data/processed_data.csv", sep = ",")
   print("Load & preprocessing --- DONE")
   
   # drop tweet list na !!!
   df["tweet_list"] = df["tweet_clean"].str.split(" ").tolist()
   df.dropna(inplace = True) # drop 2 NA

   ### doc2vec
   # generate model of vectorization
   if 'd2v.model' not in os.listdir('../models'):
      documents = [TaggedDocument (doc, [i]) for i, doc in enumerate(df["tweet_list"])]
      model = Doc2Vec (documents, vector_size=5, window=2, min_count=1)
      model.save("../models/d2v.model")
   else:
      model = Doc2Vec.load("../models/d2v.model")
   print("Doc2Vec model --- DONE")
   
   ### vectorization
   # vettore del 1o documento
   vector = model.infer_vector(df["tweet_list"].loc[0])
   # vettori per tutti i documenti
   df["vectors"] = df["tweet_list"].apply(lambda x: model.infer_vector(x))
   print("Vectorization --- DONE")
   
   # text classfication/clustering

