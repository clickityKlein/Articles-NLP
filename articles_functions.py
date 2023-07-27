# -*- coding: utf-8 -*-
"""
Functions created for the 'articles' analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC


'''
Load and Clean Data

Functions:
    - load_data
    - clean_words
    - variable_formatting
    - word_count
    - clean_data
'''
# function to load the initial dataset (assumes combined articles)
def load_data(articles_filepaths = ['data/articles1.csv',
                                    'data/articles2.csv',
                                    'data/articles3.csv',
                                    ]):
    '''
    INPUT:
        - articles_filepaths: filepath to the articles csv data
    
    OUTPUT:
        - df: DataFrame of the articles
    '''
    df1 = pd.read_csv(articles_filepaths[0])
    df2 = pd.read_csv(articles_filepaths[1])
    df3 = pd.read_csv(articles_filepaths[2])
    df = pd.concat([df1, df2, df3])
    
    return df

# function to "clean" strings: lower case, no punctuation, clean spaces
def clean_words(text):
    '''
    INPUT:
        - text: title or content of an article
        
    OUTPUT:
        - clean_text: message after the tokenization process, returned as a list
    '''
    
    # remove punctuation, lower case, and remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    clean_text = ' '.join(text.lower().strip().split())
    
    return clean_text

# function to format text strings into categorical variable format
def variable_formatting(text):
    '''
    INPUT:
        - text: string that will be turned into a callable variable
        
    OUTPUT:
        - variable_text: string that works as a callable variable
    '''
    variable_text = '_'.join(text.lower().split())
    
    return variable_text

# function to count number of words in the article
def word_count(text):
    '''
    INPUT:
        - text: cleaned string whose words will be counted
        
    OUTPUT:
        - word_count: number of words in string
    '''
    word_count = len(text.split())
    
    return word_count

# function which provides a clean dataset
def clean_data(df):
    '''
    INPUT:
        - df: DataFrame containing merged data from load_data
        
    OUTPUT:
        - df: DataFrame of cleaned data
    '''
    # drop unnecessary columns
    df.drop(['Unnamed: 0', 'id', 'author', 'url', 'year', 'month'],
            axis = 1, inplace = True)
    
    # set date column as datetime type variable
    df.date = pd.to_datetime(df.date)
    
    # drop any rows containing null values
    df.dropna(how='any', axis=0, inplace=True)
    
    # clean title and content
    df['title'] = df.title.map(clean_words)
    df['content'] = df.content.map(clean_words)
    
    # prepare publication column for variable transfer
    df['publication'] = df.publication.map(variable_formatting)
    
    # create new column for word count of content
    df['word_count'] = df.content.map(word_count)
    
    # fiilter out any articles with content less than 30 words
    # this will remove erronous data
    # filter out the largest article (entire james comey testimony)
    df = df[(df.word_count >= 30) & (df.word_count < 50000)]
    
    # reset index
    df.reset_index(drop=True, inplace=True)
    
    return df


'''
Model Creation
'''

# tokenize, remove stop words, and then lemmatize (for pipeline)
def tokenize(text):
    '''
    INPUT:
        - text: a string of text, either title or content

    OUTPUT:
        - clean_tokens: text after the tokenization process, returned as a list
    '''
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

# svc bayessearch model build
def build_model_SVC():
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('svc', SVC()),
        ])

    # parameters
    params = {
        'C': (1e-6, 100.0, 'log-uniform'),
        'gamma': (1e-6, 100.0, 'log-uniform'),
        'degree': (1, 5),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }

    # set cv
    cv = RepeatedStratifiedKFold(n_splits=10,
                                 n_repeats=3,
                                 random_state=1)

    # define evaluation
    search = BayesSearchCV(
        pipe,
        search_spaces = params,
        n_jobs = -1,
        cv=cv
        )
    
    return search

# ada & gb bayessearch model build
def build_model_double_proc():
    ada_search = {
        'model': [AdaBoostRegressor()],
        'model__learning_rate': Real(0.005, 0.9, prior="log-uniform"),
        'model__n_estimators': Integer(1, 1000),
        'model__loss': Categorical(['linear', 'square', 'exponential'])
    }
    gb_search = {
        'model': [GradientBoostingRegressor()],
        'model__learning_rate': Real(0.005, 0.9, prior="log-uniform"),
        'model__n_estimators': Integer(1, 1000),
        'model__loss': Categorical(['ls', 'lad', 'quantile'])
    }
    
    # pipeline
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', GradientBoostingRegressor())
        ])
    
    # bayessearchcv
    opt = BayesSearchCV(
        pipe,
        [(ada_search, 100), (gb_search, 100)],
        cv=5
        )
    
    return opt

# random forest classifier model build
def build_model_forest():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    # parameters
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy', 'log_Loss'],
        'clf__estimator__n_estimators': [50, 100, 200]
        }
    
    # optimize
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, cv=2)
    
    return cv
    
    

