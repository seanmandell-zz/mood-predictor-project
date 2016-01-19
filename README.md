# Mood Predictor

## Note: This is a work in progress (1/19). Check back soon.

## Overview

### Goal of the Project

My goal was to predict people's self-reported daily moods based solely on their phone usage data. I used data from the [2010-11 MIT Friends and Family study](http://realitycommons.media.mit.edu/friendsdataset.html), in which ~200 graduate students were given Android phones that tracked them (calls, texts, bluetooth proximity to other devices, etc., all anonymized). The students also filled out daily, weekly, and monthly surveys, answering questions such as the moods they felt each day.

### Methodology

I chose this project in large part because I knew it would involve extensive feature engineering, and I thought it would be a fun challenge; at the outset, it wasn't clear how much phone usage data would reveal about people's moods. In devising potentially useful features, I looked to [existing](http://hd.media.mit.edu/tech-reports/TR-670.pdf) [literature](http://disi.unitn.it/~staiano/pubs/SLAPSP_UBICOMP12.pdf), which inspired me to extract or create features related to the students' social interactions and social networks. However, other than this general guidance, I was on my own to try different features and see what worked best. I also tried different models, settling on [Gradient Boosted Regression Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), and used GridSearch to optimize parameters.

### Findings




##

create_labels.py: Extracts daily mood data to create a labels DataFrame.

feature_engineer.py: Contains functions to engineer features.

test_models.py: The engine: creates feature-label matrix and tests models.
