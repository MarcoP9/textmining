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
C:.
│   .gitignore
│   01-Text_Exploration.ipynb
│   02-Text_Processing_Representation.ipynb
│   03-Features_extraction.ipynb
│   04-Text_Classification.ipynb
│   05-Text_Classification_Binary.ipynb
│   environment.yml
│   README.md
│   requirements.txt
│   WordCloud.ipynb
│
├───data
│   │   featured_data.csv
│   │   labeled_data.csv
│   │   mask_marco.png
│   │   processed_data.csv
│   │   trump_tweets.csv
│   │
│   └───representations
│           bag_of_words.npz
│           count_vector.npz
│           doc2vec.npy
│           tf-idf.npy
│
├───models
│   ├───base
│   │       README.md
│   │
│   ├───binary
│   │       README.md
│   │
│   └───doc2vec
│           README.md
│
└───pics
    │   sentiment.PNG
    │
    └───wordcloud
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