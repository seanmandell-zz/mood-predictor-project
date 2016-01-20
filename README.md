# Mood from Phone

## Note: This is a work in progress (1/19). Check back soon.

## Table of Contents
1. [Bullet-Point Summary](#bullet-point-summary)
2. [Overview](#overview)
3. [Why I Chose This Project](#why-i-chose-this-project)
4. [Methodology](#methodology)
5. [Step 1: Create Possible Labels](#step-1-create-possible-labels)
6. [Features I Engineered](#features-i-engineered)

## Bullet-Point Summary
* I tried to predict daily mood from phone use data.
* Doing this with a high degree of accuracy appears to be difficult!
* Through extensive feature engineering, I got a proof-of-concept model that indicates that further feature engineering may be able to yield a good model.

## Overview

My goal was to predict people's self-reported daily moods based solely on their phone usage data. I used data from the [2010-11 MIT Friends and Family study](http://realitycommons.media.mit.edu/friendsdataset.html), in which ~200 graduate students were given Android phones that tracked them (calls, texts, bluetooth proximity to other devices, etc., all anonymized). The students also filled out daily, weekly, and monthly surveys, answering questions such as the moods they felt each day.

## Why I Chose This Project

I chose this project for 3 main reasons:
* **Commercial and mental health applications.**
* **Feature engineering.** I knew this would involve extensive feature engineering, and I thought it would be a fun challenge; at the outset, it wasn't clear how much phone usage data would reveal about people's moods.
* **Network/graph theory.** I was interested in learning more about how networks can shed light on day-to-day life.

## Methodology

 In devising potentially useful features, I looked to [existing](http://hd.media.mit.edu/tech-reports/TR-670.pdf) [literature](http://disi.unitn.it/~staiano/pubs/SLAPSP_UBICOMP12.pdf), which inspired me to extract or create **features related to the students' social interactions and social networks**. However, other than this general guidance, I was on my own to try different features and see what worked best. I also tried different models, settling on [Gradient Boosted Regression Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), and used GridSearch to optimize parameters.

## Step 1: Create Possible Labels

Study participants rated how happy, sad, and productive (the last of which I think counts as a mood among ambitious grad students) they were every day on a scale from 1-7.

## Features I Engineered

The features I engineered can be split into two categories: basic and advanced. You can see both summarized in the table below.

### Advanced Features, Part I

The advanced features require some explanation. Let me walk you through an example. Say you're a participant in the study, and you exchange more texts with your spouse per day, on average, than with anyone else. Say that on a given day you exchange 3 texts with your spouse, where you normally exchange 10. Out of these figures, create 3 features: 3 (daily), 10 (daily average), and 0.3 (ratio). Without looking at the data, it seems reasonable that any/all of these could be useful indicators of mood.

Instead of creating features out of interactions with just one person, I created buckets for each participant based on with whom he/she exchanged the most texts on average. Bucket 1 = top person, Bucket 2 = 2 through 5, and so on.

I repeated this procedure for each participant, for:
* Call
* SMS
* Bluetooth proximity data (a proxy for face-to-face interactions)

I determined the size of each bucket based on what I thought made sense; given more time, I'd perform a GridSearch over different possible numbers and sizes of buckets.

### Advanced Features, Part II

I also used NetworkX to create for each participant three measures of graph centrality: degree, Eigenvector, and Eigenvector weighted. I used Bluetooth proximity data (face-to-face interactions) for this, and limited the graph to study participants.

Note that these centrality measures, like the per-day averages mentioned above, are constant for each participant throughout the study period.



## Choosing a Model



## Findings

## Things I Tried That Didn't Work

* **"De-medianed" (by participant) features**, for every existing feature. This didn't improve model performance, and it made runtime a lot longer by roughly doubling the number of features.
* **Only considering more "trustworthy" survey responses**--say, ones submitted within 3 days of the day the mood question referred to.
*

## Other Stumbling Blocks

* **Missing data**.
This one surprised me given that this is a recent academic study, but in the first half of the study there appears to be a lot of missing call, text, Bluetooth, and possibly other data. The way the mood questions were asked changed halfway through the study period, so I had to use data from the second half.
* **Cross-validation intricacies**.
The folds created by scikit-learn's cross-validation are deterministic, meaning that when you don't shuffle your data before creating folds for cross-validation--which I didn't--the folds should be the same every time. The state your data is in when the folds are created--especially how your data is sorted--can matter.

In my case, I needed the data to be sorted by participant so that participants wouldn't appear in most or all of the folds; this would be a problem because the model could "learn" individual participants while training (especially because I had some features that were constant for a given participant throughout the whole study), making it unrealistically good on the test set. (Note that sorting the data by participant would likely result in up to 4 participant [given 5-fold cross-validation] being split across multiple folds, but with this representing only a tiny share of all participants, this wasn't a big enough problem to make it worth engineering around.)




![Table](https://github.com/seanmandell/mood-predictor-project/tree/master/README-Images/table_advfeatures.png)

##

create_labels.py: Extracts daily mood data to create a labels DataFrame.

feature_engineer.py: Contains functions to engineer features.

test_models.py: The engine: creates feature-label matrix and tests models.
