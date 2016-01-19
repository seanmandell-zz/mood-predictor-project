 ******** GRIDSEARCH RESULTS *************************************************

1. Initial testing
est = GradientBoostingRegressor(n_estimators=1000)
param_grid = {'learning_rate': [0.1, 0.05, 0.02],
              'max_depth': [4, 7],
              'min_samples_leaf': [3, 5, 9, 17],
              'max_features': [1.0, 0.3, 0.1]
              }
Results: {'learning_rate': 0.02, 'max_depth': 4, 'max_features': 0.1, 'min_samples_leaf': 5}


2. Further testing
est = GradientBoostingRegressor(n_estimators=1000)
param_grid = {'learning_rate': [0.05, 0.02],
              'max_depth': [2, 3, 4],
              'min_samples_leaf': [4, 5, 7],
              'max_features': [0.05, 0.1, 0.2]
              }
Results: {'learning_rate': 0.02, 'max_depth': 4, 'max_features': 0.1, 'min_samples_leaf': 7}


3. Now tuning learning rate
est = GradientBoostingRegressor(n_estimators=5000)
param_grid = {'learning_rate': [0.005, 0.01, 0.02, 0.03],
              'max_depth': [4],
              'min_samples_leaf': [7],
              'max_features': [0.1]
              }
Results: {'learning_rate': 0.005, 'max_depth': 4, 'max_features': 0.1, 'min_samples_leaf': 7}


4. Further tuning learning rate
est = GradientBoostingRegressor(n_estimators=12000)
param_grid = {'learning_rate': [0.001, 0.003, 0.005, 0.075],
              'max_depth': [4],
              'min_samples_leaf': [7],
              'max_features': [0.1]
              }
Results: {'learning_rate': 0.003, 'max_depth': 4, 'max_features': 0.1, 'min_samples_leaf': 7}

5. Now tuning subsample (i.e., testing stochastic GBR)
est = GradientBoostingRegressor(n_estimators=12000)
param_grid = {'learning_rate': [0.003],
              'max_depth': [4],
              'min_samples_leaf': [7],
              'max_features': [0.1]
              'subsample': [0.2, 0.5, 1]
              }
Results: {'learning_rate': 0.003, 'max_depth': 4, 'max_features': 0.1, 'min_samples_leaf': 7, 'subsample': 1}
*********************************************************************







******** Grid Search Code ********************************************
NOTE: This is a sample of code for demonstration purposes. It is not meant to be run.
I ran it in a __main__ function in test_models.py.

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

X = mt.feature_label_mat.drop(POSS_LABELS+['participantID', 'date'], axis=1).values
y = mt.feature_label_mat['happy'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

param_grid = {'learning_rate': [0.003],
              'max_depth': [4],
              'min_samples_leaf': [7],
              'max_features': [0.1],
              'subsample': [0.2, 0.5, 1]
              }

est = GradientBoostingRegressor(n_estimators=12000)
gs_cv = GridSearchCV(est, param_grid, n_jobs=-1, verbose=5).fit(X_train, y_train)
gs_cv.best_params_
