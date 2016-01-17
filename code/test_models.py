import numpy as np
import pandas as pd
from pandas import Timestamp
from pandas.tseries.offsets import *
from datetime import datetime
from sklearn import metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, \
                             GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from create_labels import create_poss_labels
from feature_engineer import FeatureEngineer#engineer, _daily_stats_most_freq
import networkx as nx

class ModelTester(object):
    def __init__(self, feature_text_files, poss_labels, to_dummyize, basic_features=True, \
                 advanced_call_sms_bt_features=True, other_features=True, add_centrality_chars=True, \
                 very_cutoff_inclusive=6, very_un_cutoff_inclusive=2, min_date='2010-11-12', \
                 max_date='2011-05-21', create_demedianed=False, Fri_weekend=True, keep_dow=True):
        '''
        INPUT: list of strings, list of strings, int, int
            - feature_text_files: names of CSV files containing features data
            - to_dummyize: names of moods (happy, stressed, productive) to create dummies for
            - very_cutoff_inclusive: min number for the "very" dummies to be set to 1
            - very_un_cutoff_inclusive: max number for the "very_un" dummies to be set to 1
        OUTPUT: None

        Class constructor.
        Creates all labels, each of which the model will individually attempt to predict.
        Reads CSV files specified by feature_text_files as DataFrames.
        '''
        self.poss_labels = poss_labels
        self.basic_features = basic_features
        self.advanced_call_sms_bt_features = advanced_call_sms_bt_features
        self.other_features = other_features
        self.add_centrality_chars = add_centrality_chars
        self.min_date = min_date
        self.max_date = max_date
        self.create_demedianed = create_demedianed
        self.Fri_weekend = Fri_weekend
        self.keep_dow = keep_dow

        self.feature_dfs = {}
        self.df_labels = create_poss_labels('SurveyFromPhone.csv', to_dummyize, \
                                            very_cutoff_inclusive, very_un_cutoff_inclusive)
        print "Labels created"
        self.feature_label_mat = None
        self.models = {}
        self.X_train_folds, self.X_test_folds, self.y_all_train_folds, self.y_all_test_folds = [], [], [], []
        self.n_folds = None
        self.features_used = None
        self.feature_importances = []

        ''' Reads in raw feature_dfs'''
        for text_file in feature_text_files:
            input_name = '../data/' + text_file
            df_name = "df_" + text_file.split('.')[0]
            self.feature_dfs[df_name] = pd.read_csv(input_name)
        print "Feature dfs read in"

    def _limit_dates(self):
        '''
        INPUT: String, string
        OUTPUT: None

        Keeps observations within [min_date, max_date], inclusive (where a day is defined as 4 AM to 4 AM the next day).
        Does other minimal cleaning.
        (Can currently handle bt, battery, sms, call)
        '''

        for df_name in self.feature_dfs.iterkeys():
            df = self.feature_dfs[df_name]
            if df_name == 'df_BluetoothProximity':
                ''' Limits dates to relevant period; removes possibly erroneous nighttime observations'''
                df = df.rename(columns={'date': 'local_time'})
                df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
                df = df[df['local_time'].dt.hour >= 7] # Per Friends and Family paper (8.2.1), removes b/n midnight and 7 AM
            elif df_name == 'df_Battery':
                df = df.rename(columns={'date': 'local_time'})
            elif df_name == 'df_AppRunning':
                df = df.rename(columns={'scantime': 'local_time'})

            df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
            df.loc[df['local_time'].dt.hour < 4, 'local_time'] = (pd.DatetimeIndex(df[df['local_time'].dt.hour < 4]['local_time']) - \
                                                                 DateOffset(1))
            df['date'] = df['local_time'].dt.date
            df = df.drop('local_time', axis=1)
            df = df[((df['date'] >= datetime.date(pd.to_datetime(self.min_date))) & \
                     (df['date'] <= datetime.date(pd.to_datetime(self.max_date))))]
            self.feature_dfs[df_name] = df

    def _fill_na(self):
        '''
        INPUT: None
        OUTPUT: None

        Fills in missing values: according to fillna_dict, sets to 0 or to each participant's median value.
        '''

        fillna_dict = {'df_CallLog': 'zero', 'df_SMSLog': 'zero', 'df_network': 'zero', \
                       'df_Battery': 'partic_median', 'df_BluetoothProximity': 'partic_median'}

        for df_name in self.feature_dfs.keys():
            cols = list(self.feature_dfs[df_name].columns.values)
            if fillna_dict[df_name] == 'zero':
                cols.remove('participantID')
                cols.remove('date')
                for col in cols:
                    self.feature_label_mat[col].fillna(0, inplace=True)
            elif fillna_dict[df_name] == 'partic_median':
                for col in cols:
                    if (col != 'date' and col != 'participantID'):
                        median_dict = dict(self.feature_label_mat.groupby('participantID')[col].median())
                        self.feature_label_mat.loc[pd.isnull(self.feature_label_mat[col]), col] = \
                                                            self.feature_label_mat[col].map(median_dict)



    def _create_demedianed_cols(self):
        '''
        INPUT: None
        OUTPUT: None
        Creates new "de-medianed" feature columns using each participant's median for each existing
        '''
        all_cols = list(self.feature_label_mat.columns.values)
        cols_to_remove = self.poss_labels + ['participantID', 'date']
        for col in cols_to_remove:
            all_cols.remove(col)
        feature_cols = all_cols

        for col in feature_cols:
            new_col_name = col + "_demedianed"
            df_median_by_partic = pd.DataFrame(self.feature_label_mat.groupby('participantID')[col].median()).reset_index()
            df_median_by_partic.rename(columns={col: 'median'}, inplace=True)
            median_series = self.feature_label_mat.merge(df_median_by_partic, how='left', on='participantID')['median']
            self.feature_label_mat[new_col_name] = median_series
            self.feature_label_mat[new_col_name] = self.feature_label_mat[col] - self.feature_label_mat[new_col_name]

    def _add_weekend_col(self):
        '''
        INPUT: bool, bool
        OUTPUT: None
        - Adds a dummy column to feature_label_mat called 'weekend',
        1 if the day of week is Sat/Sun (plus Fri if Fri_weekend=1), 0 otherwise
        - Also keeps day of week if keep_dow is True
        '''
        self.feature_label_mat.loc[:, 'day_of_week'] = self.feature_label_mat['date'].map(lambda x: x.dayofweek)
        day_to_split = 5 - 1 * self.Fri_weekend
        self.feature_label_mat.loc[self.feature_label_mat['day_of_week'] >= day_to_split, 'weekend'] = 1
        self.feature_label_mat.loc[self.feature_label_mat['day_of_week'] < day_to_split, 'weekend'] = 0
        if not self.keep_dow:
            self.feature_label_mat.drop('day_of_week', axis=1, inplace=True)

    def create_feature_label_mat(self):
        '''
        INPUT: list of strings, int, bool, bool
        OUTPUT: None
        Creates a feature-matrix DataFrame, and deals with missing values.
        '''
        self._limit_dates()
        ''' Engineers features'''
        # self.feature_dfs['df_network'] = engineer_bt_network(self.feature_dfs['df_BluetoothProximity'])


        for df_name, df in self.feature_dfs.iteritems():
            if self.basic_features:
                #(self, df, df_name, advanced=False, add_centrality_chars=False)
                fe = FeatureEngineer(df, df_name)
                self.feature_dfs[df_name] = fe.engineer()
            if self.advanced_call_sms_bt_features:   # Available for CallLog, SMSLog, BluetoothProximity
                if (df_name == 'df_CallLog' or df_name == 'df_SMSLog' or df_name == 'df_BluetoothProximity'):
                    if self.add_centrality_chars and df_name == 'df_BluetoothProximity':
                        fe = FeatureEngineer(df, df_name, advanced=True, add_centrality_chars=True)
                    else:
                        fe = FeatureEngineer(df, df_name, advanced=True)
                    df_newname = df_name + '_advanced'
                    self.feature_dfs[df_newname] = fe.engineer()


        ''' WILL PROBABLY DELETE MOST/ALL OF THIS BLOCK ***********************************************'''
        # df_bt = self.feature_dfs['df_BluetoothProximity']
        # # df_advanced_bt_input = df_bt[pd.notnull(df_bt['participantID.B'])]
        # # self.feature_dfs['df_Bluetooth_adv'] = _daily_stats_most_freq(df_advanced_bt_input, 'bt', bidirectional=True, \
        # #                                 partic_name='participantID', target_name='participantID.B', add_centrality_chars=True)
        # self.feature_dfs['df_CallLog_adv'] = _daily_stats_most_freq(self.feature_dfs['df_CallLog'], 'call', bidirectional=False, \
        #                                 partic_name='participantID.A', target_name='number.hash', add_centrality_chars=False)
        #
        # if self.basic_features:
        #     for name, feature_df in self.feature_dfs.iteritems():
        #         self.feature_dfs[name] = engineer(name, feature_df, basic_features, \
        #                                           advanced_call_sms_bt_features, other_features)
        ''' ******************************************************************************************* '''




        ''' Merges features and labels into one DataFrame'''
        for feature_df in self.feature_dfs.itervalues():
            self.df_labels = self.df_labels.merge(feature_df, how='left', on=['participantID', 'date'])
        self.feature_label_mat = self.df_labels

        mt.feature_label_mat = mt.feature_label_mat[pd.notnull(mt.feature_label_mat['participantID'])]
        self._fill_na()


        if list(self.feature_label_mat.columns).count('cnt nan') > 0:   #Drops 'cnt nan' column if it exists
            self.feature_label_mat.drop('cnt nan', axis=1, inplace=True)

        if self.create_demedianed:
            self._create_demedianed_cols()
        self.feature_label_mat.fillna(0, inplace=True)

        ''' Adds a dummy 'weekend', 1 for Sat/Sun (and Fri if Fri_weekend=True), 0 otherwise '''
        self._add_weekend_col()


        if list(self.feature_label_mat.columns).count('index') > 0:    #Drops 'index' column if it exists
            self.feature_label_mat.drop('index', axis=1, inplace=True)


    ''' Note to Giovanna: I don't think create_cv_pipeline is necessary any longer,
    and plan on removing it this weekend. '''
    def create_cv_pipeline(self, n_folds):
        '''
        INPUT: int
        OUTPUT: None

        Divides feature-label matrix into n_folds folds, saving each to, respectively,
        X_train_folds, X_test_folds, y_all_train_folds, and y_all_test_folds.
        To be used in n_folds-fold cross-validation.
        '''

        ''' 1. Pulls out X, y_all (y_all columns include all possible labels) '''
        self.n_folds = n_folds
        n_elems = self.feature_label_mat.shape[0]
        kf = cross_validation.KFold(n_elems, n_folds=n_folds)
        drop_from_X = self.poss_labels + ['participantID', 'date']
        self.features_used = self.feature_label_mat.drop(drop_from_X, axis=1).columns.values
        X = self.feature_label_mat.drop(drop_from_X, axis=1).values  # Need to use .values for KFold
        y_all = self.feature_label_mat[self.poss_labels].values   # Need to use .values for KFold

        ''' 2. Defines folds and saves to lists'''
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_all_train, y_all_test = y_all[train_index], y_all[test_index]
            self.X_train_folds.append(X_train)
            self.X_test_folds.append(X_test)
            self.y_all_train_folds.append(y_all_train)
            self.y_all_test_folds.append(y_all_test)
        print "Cross-validation pipeline created"

    def fit_score_models(self, models):
        '''
        INPUT: dict of model --> string (e.g.,: {rfr: 'Random Forest Regressor', ...})
        OUTPUT: None

        Fits and scores inputted models, printing out k-fold scores and average score.
        Saves feature importances in feature_importances.
        '''
        self.models = models    # Mostly to save for future reference
        for model, descrip in models.iteritems():
            mean_scores_by_label = {}
            for poss_label_col_num, poss_label in enumerate(self.poss_labels):
                scores = np.zeros(self.n_folds)
                for i in xrange(self.n_folds):
                    X_train = self.X_train_folds[i]
                    y_train = self.y_all_train_folds[i][:, poss_label_col_num]
                    model.fit(X_train, y_train)
                    scores[i] = model.score(self.X_test_folds[i], self.y_all_test_folds[i][:, poss_label_col_num])
                print "scores: ", scores
                mean_scores_by_label[poss_label] = np.mean(scores)

                ''' Feature importances '''
                importances = np.array(zip(self.features_used, model.feature_importances_))
                descending_importance_indexes = np.argsort(model.feature_importances_)[::-1]
                self.feature_importances.append((descrip, poss_label, importances[descending_importance_indexes]))

            print "\n\n", descrip
            print "==================================================="
            for label, score in mean_scores_by_label.iteritems():
                print label, " prediction score (regr-->R^2, classifier-->accur.): ", score
            print "==================================================="
            print "\n"

if __name__ == '__main__':
    ''' NOTE:
    To run, may need to manually install the latest version of networkx:
    Download the tar.gz or zip file from https://pypi.python.org/pypi/networkx/
    (As of 1/15/2016: pip install may upgrade you to version 1.10, but you need 1.11)
    '''

    '''
    [[NOTE TO SELF: PROVIDE BOOL TO NOT USE GRAPH FEATURES IN CASE OF ABOVE ISSUE]]
    '''


    ''' 1. FIELDS TO POTENTIALLY MODIFY ************************************* '''
    basic_features = True
    advanced_call_sms_bt_features = True    # Whether to include advanced Call/SMS/Bluetooth features
    #other_features = True   # Whether to include features not related to Call/SMS/Bluetooth (eg, Battery)
    N_FOLDS = 5   # Number of folds to use in cross-validation
    TO_DUMMYIZE = ['happy']    # Mood(s) to create dummies with: happy, stressed, and/or productive
    FEATURE_TEXT_FILES = [
                          "SMSLog.csv",
                          "CallLog.csv",
                          "Battery.csv",
                          "BluetoothProximity.csv"
                          ]



    ''' FINISH THIS ************************************************************************** '''
    ''' MODEL TESTER INIT: '''
    # (self, feature_text_files, poss_labels, to_dummyize, basic_features=True, \
    #          advanced_call_sms_bt_features=True, other_features=True, very_cutoff_inclusive=6, very_un_cutoff_inclusive=2, \
    #          min_date='2010-11-12', max_date='2011-05-21', create_demedianed=False, Fri_weekend=True, keep_dow=True)

    ''' *********************************************************************************** '''







    ''' Defines models '''
    ''' Regressors '''
    rfr = RandomForestRegressor(n_jobs=-1, random_state=42)
    dtr = DecisionTreeRegressor(max_depth=10)
    abr25 = AdaBoostRegressor(n_estimators=25)
    abr50 = AdaBoostRegressor(n_estimators=50) # Default
    abr100 = AdaBoostRegressor(n_estimators=100)
    abr50_squareloss = AdaBoostRegressor(n_estimators=50, loss='square')
    abr50_exploss = AdaBoostRegressor(n_estimators=50, loss='exponential')
    gbr = GradientBoostingRegressor()
    gbr_stoch = GradientBoostingRegressor(subsample=0.1) # Default n_estimators (100) much better than 500
    ''' Classifiers '''
    rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
    gbc = GradientBoostingClassifier()

    MODELS_TO_USE = [   # Which models to test. Scroll to bottom for descriptions of each
            #   rfr,
            #   dtr,
            #   abr25,
            #   abr50,
            #   abr100,
            #   abr50_squareloss,
            #   abr50_exploss,
              gbr#,
            #   gbr_stoch,
              #rfc,
              #gbc
            ]
    ''' ********************************************************************* '''

    ''' 2. FIELDS TO PROBABLY LEAVE ALONE *********************************** '''
    ''' Files to use as features'''

    POSS_LABELS = ['happy', 'stressed', 'productive']
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
    descrips_all[rfr] = 'rfr -- Random Forest Regressor'
    descrips_all[dtr] = 'dtr -- Decision Tree Regressor'
    descrips_all[abr25] = 'abr25 -- AdaBoost Regressor, 25 estimators'
    descrips_all[abr50] = 'abr50 -- AdaBoost Regressor, 50 estimators (default)'
    descrips_all[abr50_squareloss] = 'abr50_squareloss -- AdaBoost Regressor, 50 estimators (default), square loss fn'
    descrips_all[abr50_exploss] = 'abr50_exploss -- AdaBoost Regressor, 50 estimators (default), exponential loss fn'
    descrips_all[gbr] = 'gbr -- Gradient-Boosting Regressor'
    descrips_all[gbr_stoch] = 'gbr_stoch -- *stochastic* Gradient-Boosting Regressor'
    ''' Classifiers '''
    descrips_all[rfc] = 'rfc -- Random Forest Classifier'
    descrips_all[gbc] = 'gbc -- Gradient Boosting Classifier'

    model_descrip_dict = {}
    for model in MODELS_TO_USE:
        model_descrip_dict[model] = descrips_all[model]
    ''' ********************************************************************* '''


    ''' 3. Runs the model tester ******************************************** '''
    mt = ModelTester(FEATURE_TEXT_FILES, POSS_LABELS, TO_DUMMYIZE, basic_features, \
                     advanced_call_sms_bt_features)
    mt.create_feature_label_mat()
    mt.create_cv_pipeline(N_FOLDS)
    mt.fit_score_models(model_descrip_dict)

    ''' ********************************************************************* '''

    # ''' Example; may not want to use this method, though '''
    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')
