# TextMining Project

The project is written with Python 3.7.9. Several libraries are used and the whole list is present in the file _requirements.txt_.

Also, in order to avoid version dependences problems, we've used a common anaconda environment that is specified in the file _environment.yml_.

## Before execute

There are two ways to install the correct libraries:

1. with anaconda environment specific

        conda env create -f environment.yml
2. with pip library installation:

        pip install requirements.txt 

## Folder structure
```
|   .gitignore
|   01-Text_Exploration.ipynb
|   02-Text_Processing_Representation.ipynb
|   03-Features_extraction.ipynb
|   04-Text_Classification.ipynb
|   05-Text_Classification_Binary.ipynb
|   environment.yml
|   README.md
|   requirements.txt
|   tree.txt
|   WordCloud.ipynb
|   
+---.ipynb_checkpoints
|       01-Text_Exploration-checkpoint.ipynb
|       02-Text_Processing_Representation-checkpoint.ipynb
|       03-Features_extraction-checkpoint.ipynb
|       04-Text_Classification-checkpoint.ipynb
|       05-Text_Classification_Binary-checkpoint.ipynb
|       Exploration-checkpoint.ipynb
|       Lemmatization-checkpoint.ipynb
|       WordCloud-checkpoint.ipynb
|       
+---code
|   |   features.py
|   |   github_classifier.ipynb
|   |   main.py
|   |   process_text.py
|   |   Text_mining.ipynb
|   |   Text_mining_01.ipynb
|   |   utils.py
|   |   __init__.py
|   |   
|   +---.ipynb_checkpoints
|   |       github_classifier-checkpoint.ipynb
|   |       Text_mining-checkpoint.ipynb
|   |       Text_mining_01-checkpoint.ipynb
|   |       
|   \---__pycache__
|           features.cpython-37.pyc
|           process_text.cpython-37.pyc
|           utils.cpython-37.pyc
|           __init__.cpython-37.pyc
|           
+---data
|   |   featured_data.csv
|   |   labeled_data.csv
|   |   mask_marco.png
|   |   processed_data.csv
|   |   trump_tweets.csv
|   |   
|   \---representations
|           bag_of_words.npz
|           count_vector.npz
|           doc2vec.npy
|           tf-idf.npy
|           
+---models
|   +---base
|   |       adaboost.sav
|   |       best_nn.h5
|   |       svm.sav
|   |       
|   +---binary
|   |       adaboost.sav
|   |       logistic.sav
|   |       nn_3.h5
|   |       svm.sav
|   |       
|   \---doc2vec
\---pics
    |   emoji.PNG
    |   num_exc.PNG
    |   sentiment.PNG
    |   
    \---wordcloud
            hate.PNG
            hate_offensive.PNG
            neither.PNG
            not_hate.PNG
            offensive.PNG
```

## Execute
The execution follows the order of notebooks:

```
   01-Text_Exploration.ipynb
   02-Text_Processing_Representation.ipynb
   03-Features_extraction.ipynb
   04-Text_Classification.ipynb
   05-Text_Classification_Binary.ipynb
```

Each notebook generate some output files in the execution that is used in the followings notebooks. 

This division is done for code cleanup reasons. 

**NB** In Text_Classification notebooks neural network can be changed by changing the function used.