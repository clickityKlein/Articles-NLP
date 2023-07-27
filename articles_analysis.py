# -*- coding: utf-8 -*-
"""
File created to run analysis scripts
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
Load and Clean the Data
'''
from articles_functions import load_data, clean_words, variable_formatting, word_count, clean_data
# load data    
df = load_data()
# clean data
df = clean_data(df)


'''
Create Visualizations
'''
# number of articles per publisher
# analysis
publishers = df.publication.value_counts(ascending = True)
# plotting
plt.barh(y = publishers.index, width = publishers)
plt.title('Number of Articles per Publisher')

# word count statistics per publisher
# analysis
avg_words = df.groupby('publication')['word_count'].mean().sort_values(ascending = True)
min_words = df.groupby('publication')['word_count'].min().sort_values(ascending = True)
max_words = df.groupby('publication')['word_count'].max().sort_values(ascending = True)
# plotting
plt.barh(y = avg_words.index, width = avg_words)
plt.title('Average Word Count per Publisher')
plt.barh(y = max_words.index, width = max_words)
plt.title('Maximum Word Count per Publisher')
plt.barh(y = min_words.index, width = min_words)
plt.title('Minimum Word Count per Publisher')


'''
Model Creation
'''
# load appropriate functions
from articles_functions import tokenize

# split data into training and testing sets
X = df['content']
y = df['publication']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# before testing parameters, we'll take a look at a few different default classifiers
# Random Forest Classifier
results_rfc = default_forest(X_train, X_test, y_train, y_test)

# K-Nearest-Neighbors
results_knn = default_knn(X_train, X_test, y_train, y_test)

# Naive Bayes
results_nb = default_nb(X_train, X_test, y_train, y_test)

# Support Vector Classification
results_svc = default_nb(X_train, X_test, y_train, y_test)
