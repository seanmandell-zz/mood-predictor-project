# Mood from Phone

## Highlights in This README
I think that after you read either the [Very Short Summary](#a-very-short-summary) or the [Overview](#overview), two of the most interesting parts of this README are:
- [Advanced Features, Part I](#advanced-features-part-1)
- [Future Work](#future-work), which gets at why I was excited about this project

## Table of Contents
1. [A Very Short Summary](#a-very-short-summary)
2. [How to Run My Code](#how-to-run-my-code)
3. [Overview](#overview)
4. [Methodology](#methodology)
  - [Step 1: Create Possible Labels](#step-1-create-possible-labels)
  - [Step 2: Engineer Features](#step-2-engineer-features)
  - [Step 3: Choose a Model](#step-3-choose-a-model)
  - [Step 4: GridSearch](#step-4-gridsearch)
5. [Findings](#findings)
6. [Future Work](#future-work)
7. [Things I Learned](#things-i-learned)
8. [Conclusion](#conclusion)

## A Very Short Summary
* I tried to predict daily mood from phone use data.
* Doing this with a high degree of accuracy appears to be difficult!
* Through extensive feature engineering, I got a proof-of-concept model that indicates that further feature engineering may be able to yield a good model.
* (Check out my ideas for further feature engineering [down here](#future-work).)

## How to Run My Code

If you'd like to see how my program works or play around with different models, you can run my code! This is what you need to do to achieve that.

1. Download the data I used [here](http://realitycommons.media.mit.edu/friendsdataset.html). It's freely available, but you'll need to fill in some personal information.
2. Unzip/unpack the data so you have CSV files.
3. Put the files in a folder called 'data,' which should be in the same directory (folder level) as the 'code' folder.
4. Run the program from run.py. At the top are some fields you can change if you'd like; how to do so should be clearly marked.

## Overview

### Goal

My goal was to predict people's self-reported daily moods based solely on their phone usage data. I used data from the [2010-11 MIT Friends and Family study](http://realitycommons.media.mit.edu/friendsdataset.html), in which ~200 graduate students were given Android phones that tracked them (calls, texts, bluetooth proximity to other devices, etc., all anonymized). The students also filled out daily, weekly, and monthly surveys, answering questions such as the moods they felt each day.

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/happyquestion.png)

### Motivation

I chose this project for 3 main reasons:
* **Commercial and mental health applications.** Predicting mood just from phone data, besides being interesting, could be very useful.
* **Feature engineering.** I knew this would involve extensive feature engineering, and I thought it would be a fun challenge; at the outset, it wasn't clear how much phone usage data would reveal about people's moods.
* **Network/graph theory.** I was interested in learning more about how networks can shed light on day-to-day life.

## Methodology

 In devising potentially useful features, I looked to [existing](http://hd.media.mit.edu/tech-reports/TR-670.pdf) [literature](http://disi.unitn.it/~staiano/pubs/SLAPSP_UBICOMP12.pdf), which inspired me to extract or create **features related to the students' social interactions and social networks**. However, other than this general guidance, I was on my own to try different features and see what worked best. I also tried different models, settling on [Gradient Boosted Regression Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), and used GridSearch to optimize parameters.

### Step 1: Create Possible Labels

Study participants rated how happy, stressed, and productive (the last of which I think counts as a mood among ambitious grad students) they were every day on a scale from 1-7. I also created dummy versions of these; for example, a happiness of 4 or higher would be 1 for my happy dummy, 0 otherwise. In addition, I created very (happy/stressed/productive) and very un-(happy/stressed/productive) dummies, where 6 and 2 were the respective, inclusive cutoffs.

### Step 2: Engineer Features

The features I engineered can be split into two categories: basic and advanced. You can see both summarized in the table below.

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/table_featureoverview.png)

#### Advanced Features, Part I

The advanced features require some explanation. Let me walk you through an example. Say you're a participant in the study, and you exchange more texts with your spouse per day, on average, than with anyone else. Say that on a given day you exchange 3 texts with your spouse, where you normally exchange 10. Out of these figures, create 3 features: 3 (daily), 10 (daily average), and 0.3 (ratio). These 3 features each get at slightly different things, so any or all of them seem like they could be useful.

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/advfeatures1.png)

Instead of creating features out of interactions with just one person, I created buckets for each participant based on with whom he/she exchanged the most texts on average. Bucket 1 = top person, Bucket 2 = 2 through 5, and so on.

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/advfeatures2.png)

I repeated this procedure for each participant, for:
* Call
* SMS
* Bluetooth proximity data (a proxy for face-to-face interactions)

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/advfeatures3.png)

I determined the size of each bucket based on what I thought made sense; given more time, I'd perform a GridSearch over different possible numbers and sizes of buckets.

#### Advanced Features, Part II

I also used NetworkX to create for each participant three measures of graph centrality: degree, Eigenvector, and Eigenvector weighted. I used Bluetooth proximity data (face-to-face interactions) for this, and limited the graph to study participants.

Note that these centrality measures, like the per-day averages mentioned above, are constant for each participant throughout the study period.

### Step 3: Choose a Model

I tried running various sets of features through various models to see what performed best. For example, I tried:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Machine Regressor (both rbf and polynomial kernel)
- AdaBoost Regressor (varying learning rates and loss functions)
- Gradient Boosted Regressor (including stochastic)

I tried to predict 'happy,' 'stressed,' and 'productive,' levels (from 1 to 7) with all of these.

I also tried a few classifier models the dummy versions of these variables (e.g., happy or not, very happy or not, very unhappy or not). However, I didn't pursue classification very much, because (a) I was more interested in making specific number predictions, and (b) the classifiers didn't seem to be obvious improvements over the regressors. (Side note: an interesting extension of this project would be to try to predict, e.g., very unhappy; this could have mental health applications.)

After I created all my features, I looked at R^2 values for different regressor models for each of the three moods. See the below graph.

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/choosing_model.png)

0.0 is a good baseline to compare the models with, because that's the R^2 associated with always simply guessing the average mood over the whole study. As you can see, a couple of the models fared pretty poorly. I played around with the 2 boosting models plus the SVM, ultimately getting the best results with the gradient boosted regression trees (GBRT).

GBRT is often a pretty effective off-the-shelf machine learning model. It trains by sequentially fitting many [decision trees](https://en.wikipedia.org/wiki/Decision_tree) to the previous decision tree's residuals, then outputting a model that averages out all of the decision trees.

### Step 4: GridSearch

I used scikit-learn's GridSearchCV to optimize the GBRT's hyperparameters. You can read more about the GridSearch iterations [here](https://github.com/seanmandell/mood-predictor-project/blob/master/gridsearch_results.md). In short: I ended up using a learning rate of 0.003 (very slow), 12,000 estimators, max depth of 4, max features (at each split, a la random forest) of 0.1, and minimum samples per leaf of 7.

## Findings

Did all the feature engineering pay off? To decide, we can look at how adjusted R^2 changes as we add in more features. (Unadjusted R^2 always increases with more features, hence the use of the adjusted metric.)

![Table](https://raw.githubusercontent.com/seanmandell/mood-predictor-project/master/README-Images/feat_engine_work.png)

We move left to right as we add in more features. Adding in the advanced features helps, and adding in the measures of graph centrality on top of that are a slight improvement. However, these adjusted R^2 values are pretty low. On one hand, this is disappointing, but I view this project as more a proof of concept than a finished model. Engineering advanced features brings the model up from slightly worse than guessing the mean to a bit better than that. While the gains aren't huge, I think it's possible that additional feature engineering could keep improving the model.

## Future Work

**More feature engineering.** As I say above, I think this model could be improved through more feature engineering. I had a lot of ideas about what features might be useful, such as:
- Length of Bluetooth face-to-face interactions, possibly interacted with time of day.
- Dummy variables for, say, the 10 people each participant interacts with the most, equal to 1 when the two were in proximity (or texted or called each other) in a given day.
- Amount of sleep at night (inferred). Phone data has been used to accurately infer people's nightly sleep; it helps that people often use their phone right before going to bed and right after waking up. Amount of sleep has been shown to be correlated with mood.
- Measures of, say, weekly graph centrality for each participant, rather than just over the whole study period.
- Along similar lines, measures of graph centrality of people participants interact with.
- The model's mood prediction for people a participant interacts with.
- Comparisons between the method(s) of sociability a participant interacts with. For example, whether someone texts a lot of people but doesn't have many face-to-face interactions.
- The number of new people (i.e., those not previously interacted with) a person interacts with (could only introduce after a certain amount of previous days' data).

The list could go on and on, and most of these features could be slightly varied in ways that would add many more features. For example, for the second idea--dummy variables for 10 commonly interacted-with people--there are many aspects of interactions with said people that could be featurized. This could easily turn into 10s or 100s of additional features.

## Things I Learned

### Things I Tried That Didn't Work

* **"De-medianed" (by participant) features**, for every existing feature. This didn't improve model performance, and it made runtime a lot longer by roughly doubling the number of features.
* **Only considering more "trustworthy" survey responses**--say, ones submitted within 3 days of the day the mood question referred to.

### Stumbling Blocks

* **Missing data**.
This one surprised me given that this is a recent academic study, but in the first half of the study there appears to be a lot of missing call, text, Bluetooth, and possibly other data, at least in the publicly available dataset. The way the mood questions were asked changed halfway through the study period, so I had to use data from the second half.

* **Cross-validation intricacies**.
The folds created by scikit-learn's cross-validation are deterministic, meaning that when you don't shuffle your data before creating folds for cross-validation--which I didn't--the folds should be the same every time. The state your data is in when the folds are created--especially how your data is sorted--can matter.

In my case, I needed the data to be sorted by participant so that participants wouldn't appear in most or all of the folds; this would be a problem because the model could "learn" individual participants while training (especially because I had some features that were constant for a given participant throughout the whole study), making it unrealistically good on the test set. (Note that sorting the data by participant would likely result in up to 4 participant [given 5-fold cross-validation] being split across multiple folds, but with this representing only a tiny share of all participants, this wasn't a big enough problem to make it worth engineering around.)

At first, when I got a preliminary model up and running, I mistakenly thought that my data was sorted by participant, when it was actually sorted by date. This led me to believe my model was much better than it was! The fact that the model performed better in this cross-validation set up, when it could "learn" individual participants, implies that certain participants were happier on average than others during the study period, and that the differences were big enough to improve the model's performance by a significant amount.

## Conclusion

While I wish my model had performed better (who doesn't?), I think it was a good experience for me to try hard and have a bit of success but struggle to get as far as I wanted. It forced me to think outside of the box and, while I didn't have time to implement [all of my ideas](#future-work), in the future I'll be better at attacking open-ended problems and knowing where to look to refine my model.
