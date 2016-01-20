from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
                             GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from test_models import ModelTester


if __name__ == '__main__':
    ''' NOTE:
    To run, may need to manually install the latest version of networkx:
    Download the tar.gz or zip file from https://pypi.python.org/pypi/networkx/
    (As of 1/15/2016: pip install may upgrade you to version 1.10, but you need 1.11)
    Alternatively, you can set add_centrality_chars to False, so NetworkX will not
    be needed.
    '''

    ''' 1. FIELDS TO POTENTIALLY MODIFY ************************************* '''
    basic_features = True   # Whether to include basic features for all dfs
    advanced_call_sms_bt_features = True    # Whether to include advanced Call/SMS/Bluetooth features
    add_centrality_chars = True     # Whether to include graph centrality characteristics (Bluetooth)
    N_FOLDS = 5   # Number of folds to use in cross-validation
    POSS_LABELS = ['happy', 'stressed']#, 'productive']
    TO_DUMMYIZE = ['happy']    # Mood(s) to create dummies with: happy, stressed, and/or productive
    FEATURE_TEXT_FILES = [
                          "SMSLog.csv",
                          "CallLog.csv",
                          "Battery.csv",
                          "BluetoothProximity.csv"
                          ]

    ''' Defines models '''
    svr = SVR()
    svr_poly = SVR(kernel='poly')
    lr = LinearRegression()
    rfr = RandomForestRegressor(n_jobs=-1, random_state=42)
    dtr = DecisionTreeRegressor(max_depth=10)
    abr25 = AdaBoostRegressor(n_estimators=25)
    abr50 = AdaBoostRegressor(n_estimators=50) # Default
    abr100 = AdaBoostRegressor(n_estimators=100)
    abr100_slow = AdaBoostRegressor(n_estimators=100, learning_rate=0.5)
    abr500_slow = AdaBoostRegressor(n_estimators=500, learning_rate=0.15)
    abr50_squareloss = AdaBoostRegressor(n_estimators=50, loss='square')
    abr50_exploss = AdaBoostRegressor(n_estimators=50, loss='exponential')
    gbr = GradientBoostingRegressor()
    gbr_gridsearched = GradientBoostingRegressor(n_estimators=12000, learning_rate=0.003, max_depth=4,\
                                                 max_features=0.1, min_samples_leaf=7)
    gbr_stoch = GradientBoostingRegressor(subsample=0.1)

    ''' Note: if using a model (e.g., Linear Regression) that doesn't support feature importances,
              must comment out 3 lines near the end ofof test_models.py under the comment 'Feature Importances'
    '''
    MODELS_TO_USE = [   # Which models to test. Scroll to bottom for descriptions of each
            #   rfr,
            #   dtr,
            #   abr25,
              abr50,
            #   abr100,
            #   abr50_squareloss,
            #   abr50_exploss,
              gbr#,
                # gbr_gridsearched,
            #   gbr_stoch,
                # svr,
                # lr,
                # svr_poly,
                # abr100_slow,
                # abr500_slow
            ]
    ''' ********************************************************************* '''




    ''' 2. FIELDS TO PROBABLY LEAVE ALONE *********************************** '''

    MIN_DATE = '2010-11-12'
    MAX_DATE = '2011-05-21'

    for label in TO_DUMMYIZE:
        dummy_name = label + '_dummy'
        very_name = 'very_' + label
        very_un_name = 'very_un' + label
        POSS_LABELS += [dummy_name, very_name, very_un_name]

    ''' Loads up {model-->description} dictionary to pass into fit_score_models '''
    descrips_all = {}
    ''' Regressors '''
    descrips_all[svr] = 'svr -- Support Vector Machine Regressor'
    descrips_all[svr_poly] = 'svr_poly -- Support Vector Machine Regressor, polynomial kernel'
    descrips_all[lr] = 'lr -- Linear Regression'
    descrips_all[rfr] = 'rfr -- Random Forest Regressor'
    descrips_all[dtr] = 'dtr -- Decision Tree Regressor'
    descrips_all[abr25] = 'abr25 -- AdaBoost Regressor, 25 estimators'
    descrips_all[abr50] = 'abr50 -- AdaBoost Regressor, 50 estimators (default)'
    descrips_all[abr50_squareloss] = 'abr50_squareloss -- AdaBoost Regressor, 50 estimators (default), square loss fn'
    descrips_all[abr50_exploss] = 'abr50_exploss -- AdaBoost Regressor, 50 estimators (default), exponential loss fn'
    descrips_all[abr100_slow] = 'abr100_slow -- AdaBoost Regressor, 100 estimators, learning_rate=0.5'
    descrips_all[abr500_slow] = 'abr500_slow -- AdaBoost Regressor, 500 estimators, learning_rate=0.15'
    descrips_all[gbr] = 'gbr -- Gradient-Boosting Regressor'
    descrips_all[gbr_stoch] = 'gbr_stoch -- *stochastic* Gradient-Boosting Regressor'
    descrips_all[gbr_gridsearched] = 'gbr gridsearched -- Gradient-Boosting Regressor, optimized'

    model_descrip_dict = {} # To pass in to mt.fit_score_models below
    for model in MODELS_TO_USE:
        model_descrip_dict[model] = descrips_all[model]
    ''' ********************************************************************* '''

    ''' 3. Runs the model tester ******************************************** '''
    mt = ModelTester(FEATURE_TEXT_FILES, POSS_LABELS, TO_DUMMYIZE, basic_features, \
                     advanced_call_sms_bt_features, add_centrality_chars=add_centrality_chars)
    mt.create_feature_label_mat()
    mt.create_cv_pipeline(N_FOLDS)
    mt.fit_score_models(model_descrip_dict)
    ''' ********************************************************************* '''
