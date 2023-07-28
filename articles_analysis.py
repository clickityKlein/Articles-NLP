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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostRegressor


'''
Load and Clean the Data
'''
from articles_functions import load_data, clean_words, variable_formatting, word_count, clean_data
# load data    
df = load_data()
# clean data
df = clean_data(df)


'''
Data Visualizations
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

# word count histograms
# full set
df.word_count.hist()
plt.title('Histogram of Word Count - Full Set')
plt.xlabel('Word Count')
plt.ylabel('Article Count')
# 6,000 word limit
df[df.word_count<6000].word_count.hist()
plt.title('Histogram of Word Count - 6,000 Word Limit')
plt.xlabel('Word Count')
plt.ylabel('Article Count')
# 3,000 word limit
df[df.word_count<3000].word_count.hist()
plt.title('Histogram of Word Count - 3,000 Word Limit')
plt.xlabel('Word Count')
plt.ylabel('Article Count')


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

# Naive Bayes (requires dense matrix)
results_nb = default_nb(X_train, X_test, y_train, y_test)

# Linear Support Vector Classification (requires dense matrix)
results_svc = default_nb(X_train, X_test, y_train, y_test)

# ADA Boost
results_ada = default_ada(X_train, X_test, y_train, y_test)

# now we'll try to fine tune random forest classification
search = build_model_forest()
search.fit(X_train, y_train)
search.best_params_
'''
{'clf__max_depth': 10}
'''
y_pred = search.predict(X_test)
results = classification_report(y_test, y_pred, output_dict=True)
results = pd.DataFrame(results).transpose()

# the default is random forest classifier is the best model


'''
Model Visualizations
'''
# obtain accuracies & f1-scores
classifier_models = {'Random Forest Default': results_rfc,
                     'K-NN': results_knn,
                     'Naive Bayes': results_nb,
                     'Support Vecotr': results_svc,
                     'ADA Boost': results_ada,
                     'Random Forest Tuned': results}

accs = {}
f_scores = {}
for clf in classifier_models:
    acc_idx = classifier_models[clf].index.get_loc('accuracy')
    accs[clf] = classifier_models[clf].loc['accuracy']['f1-score']
    f_scores[clf] = classifier_models[clf].iloc[:acc_idx]['f1-score']

# create plots for all models
# Random Forest Default
plt.barh(y = f_scores['Random Forest Default'].index,
         width = f_scores['Random Forest Default'])
plt.title('Random Forest Default')

# K-NN
plt.barh(y = f_scores['K-NN'].index,
         width = f_scores['K-NN'])
plt.title('K-NN')

# Naive Bayes
plt.barh(y = f_scores['Naive Bayes'].index,
         width = f_scores['Naive Bayes'])
plt.title('Naive Bayes')

# Support Vector
plt.barh(y = f_scores['Support Vector'].index,
         width = f_scores['Support Vector'])
plt.title('Support Vector')

# ADA Boost
plt.barh(y = f_scores['ADA Boost'].index,
         width = f_scores['ADA Boost'])
plt.title('ADA Boost')

# Random Forest Tuned
plt.barh(y = f_scores['Random Forest Tuned'].index,
         width = f_scores['Random Forest Tuned'])
plt.title('Random Forest Tuned')

# Accuracies of the Models
plt.bar(x = accs.keys(), height = accs.values())
plt.xticks(rotation = 90)
plt.title('Model Accuracies')

# save models as pickle files

