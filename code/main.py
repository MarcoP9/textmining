import nltk
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument # doc2vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# results
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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
      model = Doc2Vec (documents, vector_size=100, window=10, min_count=1)
      model.save("../models/d2v.model")
   else:
      model = Doc2Vec.load("../models/d2v.model")
   print("Doc2Vec model --- DONE")
   
   ### vectorization
   # vettori per tutti i documenti
   df["vectors"] = df["tweet_list"].apply(lambda x: model.infer_vector(x))
   print("Vectorization --- DONE")
   
   ### text classfication
   # prepare data
   X_data = np.stack(df["vectors"], axis = 0)
   Y_data = df["class"]
   # verifica dimensioni vettori
   assert X_data.shape[0] == Y_data.shape[0]   
   
   # split train-test
   X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data,
                                                       test_size = 0.2,
                                                       random_state = 42,
                                                       shuffle = True,
                                                       stratify = Y_data)
   
   assert X_train.shape[0] == y_train.shape[0]
   assert X_test.shape[0] == y_test.shape[0]
   
   ### training
   def c_matrix(y_val, y_pred, classes):
       cm = confusion_matrix(y_val, y_pred)
       fig, ax= plt.subplots(figsize = (8,6))
       sns.heatmap(cm, annot=True, annot_kws={"size": 10},
                  linewidths=.2, fmt="d", cmap="PuBu")
       plt.xlabel("Predicted Class", size = 12, horizontalalignment="right")
       plt.ylabel("True Class", size = 12)
       ax.set_yticklabels(classes, rotation = 45, fontdict= {'fontsize': 10})
       ax.set_xticklabels(classes, rotation = 30, fontdict= {'fontsize': 10})
       plt.title("Confusion matrix", size = 20)
       plt.show()
   
   # class weights
   weights = df['class'].value_counts() / len(df['class'])
   # Logistic regression
   #model = LogisticRegression(C = 1, random_state = 42, class_weight= {0 : weights[0],
   #                                                                    1 : weights[1],
   #                                                                    2 : weights[2]})
   model = SVC(random_state = 42, class_weight= {0 : weights[0],
                                                 1 : weights[1],
                                                 2 : weights[2]})
   
   model.fit(X_train, y_train)
   print("Modelling --- DONE")
   
   # performance on TRAIN
   y_pred = model.predict(X_train)   
   print('Classification report:')
   print(classification_report(y_train, y_pred))
   c_matrix(y_train, y_pred, ["Hate", "Offensive", "Neither"])
      
   # performance on TEST
   y_pred = model.predict(X_test)   
   print('Classification report:')
   print(classification_report(y_test, y_pred))
   c_matrix(y_test, y_pred, ["Hate", "Offensive", "Neither"])
   
   
   
   # fare analisi esplorativa. Quante parole in ogni tweet?
   # modulo chi2 di scikit per plottare unigrammi e bigrammi
   # parole negative le eliminiamo? (not, )
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
