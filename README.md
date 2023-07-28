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


**Implementation**

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