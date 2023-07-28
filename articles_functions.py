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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostRegressor
import pickle

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
def load_data(articles_filepath = 'data/articles_reduced.csv'):
    '''
    INPUT:
        - articles_filepaths: filepath to the articles csv data
    
    OUTPUT:
        - df: DataFrame of the articles
    '''
    df = pd.read_csv(articles_filepath)
    
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
    df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'id', 'author', 'url', 'year', 'month'],
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

# Random Forest Classifier
def default_forest(X_train, X_test, y_train, y_test):
    '''
    INPUT:
        - X_train: training split from dataset, article text
        - X_test: testing split from dataset, article text
        - y_train: training split from dataset, network name
        - y_test: testing split from dataset, network name
    
    OUTPUT:
        - results: DataFrame of the model's classification report
    '''
    # pipeline with default Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
        ])
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # test the classifier and calculate the results
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    results = pd.DataFrame(results).transpose()
    
    # return the classification report as a dataframe
    return results

# K-Nearest-Neighbors
def default_knn(X_train, X_test, y_train, y_test):
    '''
    INPUT:
        - X_train: training split from dataset, article text
        - X_test: testing split from dataset, article text
        - y_train: training split from dataset, network name
        - y_test: testing split from dataset, network name
    
    OUTPUT:
        - results: DataFrame of the model's classification report
    '''
    # pipeline with default K-Nearest-Neighbors
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier())
        ])
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # test the classifier and calculate the results
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    results = pd.DataFrame(results).transpose()
    
    # return the classification report as a dataframe
    return results

# Naive Bayes
def default_nb(X_train, X_test, y_train, y_test):
    '''
    INPUT:
        - X_train: training split from dataset, article text
        - X_test: testing split from dataset, article text
        - y_train: training split from dataset, network name
        - y_test: testing split from dataset, network name
    
    OUTPUT:
        - results: DataFrame of the model's classification report
    '''
    # pipeline with default Naive Bayes
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
        ])
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # test the classifier and calculate the results
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    results = pd.DataFrame(results).transpose()
    
    # return the classification report as a dataframe
    return results

# Support Vector Classification
def default_svc(X_train, X_test, y_train, y_test):
    '''
    INPUT:
        - X_train: training split from dataset, article text
        - X_test: testing split from dataset, article text
        - y_train: training split from dataset, network name
        - y_test: testing split from dataset, network name
    
    OUTPUT:
        - results: DataFrame of the model's classification report
    '''
    # pipeline with default Support Vector Classification
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
        ])
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # test the classifier and calculate the results
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    results = pd.DataFrame(results).transpose()
    
    # return the classification report as a dataframe
    return results

# ADA Boost Classification
def default_ada(X_train, X_test, y_train, y_test):
    '''
    INPUT:
        - X_train: training split from dataset, article text
        - X_test: testing split from dataset, article text
        - y_train: training split from dataset, network name
        - y_test: testing split from dataset, network name
    
    OUTPUT:
        - results: DataFrame of the model's classification report
    '''
    # pipeline with default Support Vector Classification
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', AdaBoostClassifier())
        ])
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # test the classifier and calculate the results
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    results = pd.DataFrame(results).transpose()
    
    # return the classification report as a dataframe
    return results

'''
Random Forest Classifier with parameters ran through GridSearchCV
'''
# random forest classifier model build
def build_model_forest():
    # pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
        ])
    
    # parameters
    parameters = {
        'clf__max_depth': [1, 5, 10]
        }
    
    # optimize
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, verbose=3, cv=2)
    
    return cv
