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
- Data: [Kaggle.](https://www.kaggle.com/datasets/snapcrack/all-the-news)

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
- *load_data*
- *clean_words*
- *variable_formatting*
- *word_count*
- *clean_data*
    
**Model Creation**
Functions:
- *tokenize*
- *default_forest*
- *default_knn*
- *default_nb*
- *default_svc*
- *default_ada*
- *build_model_forest*

[Table of Contents](#table-of-contents)


## Data Description

[Table of Contents](#table-of-contents)


## Methodology
**Data Preprocessing**

**Implementation**

## Analysis
**Data Exploration**

**Data Visualization**


## Results
**Model Evaluation & Validation**

**Justification**

[Table of Contents](#table-of-contents)


## Conclusion
**Reflection**

**Improvement**

[Table of Contents](#table-of-contents)