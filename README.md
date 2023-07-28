# Publisher Prediction: Natural Language Processing (NLP) with News Articles
The goal of this project was to attempt to create an NLP model which could correctly
identify the publisher given an article.

Along with supporting data and images in their respective folders, the project
uses two main files:
1. *articles_analysis.py*
2. *articles_functions.py*

**See also the medium article.**

## Table of Contents
- [Libraries Used](#libraries-used)
- [Project Definition](#project-definition)
- [Functions](#functions)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Analysis](#analysis)
- [Results](#results)
- [Conclusion](#conclusion)


## Libraries Used
The following packages are required:
- numpy
- pandas
- matplotlib.pyplot
- nltk
- re
- sklearn


[Table of Contents](#table-of-contents)


## Project Definition
**Overview**
- Domain: News / Media
- Origin: Initial Attempt, Original Problem & Solution 
- Data: [Kaggle](https://www.kaggle.com/datasets/snapcrack/all-the-news)

**Statement**

After cleaning and exploring the data via statistical and visual methods, attempt to
create a classification model which will accurately predict publisher if given a news
article.

Without higher processing power, and a more equipt dataset, this will likely not be very
accurate. Additionally, this only accounts for a small portion of the thousands of
publishers and news networks, which means any article from outside the publishers in this
dataset will result in an incorrect prediction.

**Metrics**

Our main measures of model performance will be:
- *f1-score*
- *accuracy*

*Accuracy* is generally a good starting place, however *f1-score* has a slightly
deeper and more complex meaning.

[Table of Contents](#table-of-contents)


## Functions
The following are descriptions of the functions provided in the *articles_functions.py*
file.

**Load and Clean Data**

Functions:
- *load_data*: loads the combined and reduced dataset
- *clean_words*: lowercase the characters, remove punctuation and tidy spaces
- *variable_formatting*: format publisher column for use as variables
- *word_count*: counts the number of words in a given string (used for articles)
- *clean_data*: combines many of the above functions as well as removes unnecessary columns
    
**Model Creation**

Functions:
- *tokenize*: tokenizes text (i.e. each word becomes an element in a list), removes stop words,
and then lemmatizes text (i.e. puts words in their base form)
- *default_forest*: sklearn's RandomForestClassifier with all defaults, passed through
a pipeline along with *tokenize* and a TfidfTransformer, and returns DataFrame
containing the classification report
- *default_knn*: sklearn's KNeighborsClassifier with all defaults...
- *default_nb*: sklearn's MultinomialNB with all defaults...
- *default_svc*: sklearn's LinearSVC with all defaults...
- *default_ada*: sklearn's AdaBoostClassifier with all defaults...
- *build_model_forest*: builds on the RandomForestClassifier by passing several parameters
through a GridSearchCV object

[Table of Contents](#table-of-contents)


## Data Description
The data used in this analysis was found on [Kaggle](https://www.kaggle.com/datasets/snapcrack/all-the-news).

The author of the data used a BeautifulSoup webscraper to obtain articles and associated datapoints
from several different sources. The data was uploaded in three datasets:
- artices1.csv
- artices2.csv
- artices3.csv

The datasets contains about 50,000 articles each, and have (potentially) the following columns:
- ID
- title
- publication
- author
- date
- year
- month
- url
- content

The methodology section ahead will discuss cleaning and combining of the datasets, however,
the data used in the analysis was a combination of the three files reduced to a
total of 10,000 articles.

[Table of Contents](#table-of-contents)


## Methodology
**Data Preprocessing**
The webscraper used to procure the dataset did pretty well, however there were some
areas that needed help. We'll go through each column and describe what was found.
- ID: A tracking number which really had no relevance, and was subsequently dropped.
- title: This was trickier than it first appeared. Some of the titles had their publishers in their
text, such as "title_text - publisher". Initially, the intent was to remove this and just have the
title text. However, upon further inspection, some titles would reference counterpart publishers,
or would have publisher's names in the title referencing something not related to the publisher.
For example, the publisher "Atlantic" could be self-reference, a counterpart reference, or a reference
to something like an ocean. In lieu of that information, no parts were removed from the tiles.
- publication: This was probably the cleanest column of the dataset. It featured 15 publishers,
all of which were never restated with typos.
- author: This datapoint had quite a few missing values. In the complete dataset (around 150,000 articles),
there were about 11% of authors missing. Along with the missing values, sometimes the author would be 
the publisher, an organizer, or consist of multiple authors. If it were more uniform, this could've had
more potential, however was ultimately dropped.
- date: Given the data was somewhat cherry picked and not uniform, this column didn't provide much
value and was subsequently dropped.
- year: Redundant with date, dropped.
- month Redundant with date, dropped.
- url: High number of missing values, irrelevant information, dropped.
- content: This is what we were training our model on, and thus had quite a bit of time
spent on. We removed puncuation, extra and leading/trailing spaces. After this, we counted 
the words (and tracked in the word_count column). There were quite a few articles whose content
was either blank or was scraped incorrectly. An example of an incorrect scrape that occurred frequently
was the scrape of "Advertisement". To account for this, we set a minimum word count of 30. Additionally,
there was a single article with a word count in the 50,000's which ended up being the entire 
James Comey testimony. With the next lowest having a word count in the 20,000's, we capped article
length at a word count of 50,000.

Due to size constraints in GitHub, and time/processor constraints with building an NLP model, the analysis
and model were built off a subset of the combined dataset at 10,000 articles.

In *articles_analysis*, all of the above is wittled down to the following:
```
# load data    
df = load_data()
# clean data
df = clean_data(df)
```

**Implementation**
After some exploration both statistically and visually (see Analysis section below), it was time to test models
to test some classification models. The idea was to test a few different classifiers, and then run parameters through
GridSearchCV of the best model.

The general format of the default model testing was:
```
# Classifier
def default_classifier(X_train, X_test, y_train, y_test):
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
        ('clf', Classifier())
        ])
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # test the classifier and calculate the results
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    results = pd.DataFrame(results).transpose()
    
    # return the classification report as a dataframe
    return results
```

We tested several models, ultimately running the code in *articles_analysis*:
```
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
```
See results in the Analysis and Results sections.

[Table of Contents](#table-of-contents)


## Analysis
**Data Exploration**

**Data Visualization**



[Table of Contents](#table-of-contents)


## Results
**Model Evaluation & Validation**

**Justification**

[Table of Contents](#table-of-contents)


## Conclusion
**Reflection**

**Improvement**

[Table of Contents](#table-of-contents)