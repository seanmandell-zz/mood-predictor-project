import numpy as np
import pandas as pd
from pandas import Timestamp
from pandas.tseries.offsets import *
from datetime import datetime
from sklearn import cross_validation
from create_labels import create_poss_labels
from feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA


class ModelTester(object):
    def __init__(self, feature_text_files, poss_labels, to_dummyize, basic_features=True, \
                 advanced_call_sms_bt_features=True, add_centrality_chars=True, \
                 reduce_dimensions=False, very_cutoff_inclusive=6, \
                 very_un_cutoff_inclusive=2, min_date='2010-11-12', max_date='2011-05-21', \
                 create_demedianed=False, Fri_weekend=True, keep_dow=True):
        '''
        INPUT:
            - feature_text_files: list of strings--CSV files containing features data
            - poss_labels: list of strings of dependent vars (happy, sad, and/or stressed)
            - to_dummyize: names of moods (happy, stressed, productive) to create dummies for
            - basic_features: whether to engineer basic features
            - advanced_call_sms_bt_features: whether to engineer basic features
            - add_centrality_chars: whether to add graph centrality measures for each participant
            - very_cutoff_inclusive: min number for the "very" dummies to be set to 1
            - very_un_cutoff_inclusive: max number for the "very_un" dummies to be set to 1
            - min_date: first date to include. Suggest leaving as default.
            - max_date: last date to include. Suggest leaving as default.
            - create_demedianed: whether to create "de-medianed" (by participant) feature columns
            - Fri_weekend: whether to consider Friday part of the weekend for the weekend dummy.
            - keep_dow: whether to keep dow (day of week) as a feature.
        OUTPUT: None

        Class constructor.
        Creates all labels, each of which the model will individually attempt to predict.
        Reads CSV files specified by feature_text_files as DataFrames.
        '''
        self.poss_labels = poss_labels
        self.basic_features = basic_features
        self.advanced_call_sms_bt_features = advanced_call_sms_bt_features
        self.add_centrality_chars = add_centrality_chars
        self.reduce_dimensions = reduce_dimensions
        self.min_date = min_date
        self.max_date = max_date
        self.create_demedianed = create_demedianed
        self.Fri_weekend = Fri_weekend
        self.keep_dow = keep_dow

        self.feature_dfs = {}
        self.feature_dfs_forflmat = {}  # Fully cleaned and engineered; ready for feat-lab mat
        self.df_labels = create_poss_labels('SurveyFromPhone.csv', poss_labels, to_dummyize, \
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
        INPUT: None
        OUTPUT: None

        Keeps observations within [min_date, max_date], inclusive (where a day is defined as 4 AM to 4 AM the next day).
        Does other minimal cleaning.
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

        for df_name in self.feature_dfs_forflmat.keys():
            cols = list(self.feature_dfs_forflmat[df_name].columns.values)
            for to_remove in ['index', 'cnt']:
                if cols.count(to_remove) > 0:
                    cols.remove(to_remove)
            df_name_orig = '_'.join(df_name.split('_')[:2])  # Strips off 'advanced' where applicable
            if fillna_dict[df_name_orig] == 'zero':
                cols.remove('participantID')
                cols.remove('date')
                for col in cols:
                    self.feature_label_mat[col].fillna(0, inplace=True)
            elif fillna_dict[df_name_orig] == 'partic_median':
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
        INPUT: None
        OUTPUT: None
        - Adds a dummy column to feature_label_mat called 'weekend',
        1 if the day of week is Sat/Sun (plus Fri if self.Fri_weekend=1), 0 otherwise
        - Also keeps day of week if self.keep_dow is True
        '''
        self.feature_label_mat.loc[:, 'day_of_week'] = self.feature_label_mat['date'].map(lambda x: x.dayofweek)
        day_to_split = 5 - 1 * self.Fri_weekend
        self.feature_label_mat.loc[self.feature_label_mat['day_of_week'] >= day_to_split, 'weekend'] = 1
        self.feature_label_mat.loc[self.feature_label_mat['day_of_week'] < day_to_split, 'weekend'] = 0
        if not self.keep_dow:
            self.feature_label_mat.drop('day_of_week', axis=1, inplace=True)

    def create_feature_label_mat(self):
        '''
        INPUT: None
        OUTPUT: None
        Creates a feature-matrix DataFrame, and deals with missing values.
        '''
        self._limit_dates()
        ''' Engineers features'''
        for df_name, df in self.feature_dfs.items():
            if self.advanced_call_sms_bt_features:
                df_for_adv = df.copy()
                if df_name == 'df_BluetoothProximity':
                    df_for_adv = df_for_adv[pd.notnull(df_for_adv['participantID.B'])]
            if self.basic_features:
                fe = FeatureEngineer(df, df_name)
                self.feature_dfs_forflmat[df_name] = fe.engineer()
            if self.advanced_call_sms_bt_features:   # Available for CallLog, SMSLog, BluetoothProximity
                if (df_name == 'df_CallLog' or df_name == 'df_SMSLog' or df_name == 'df_BluetoothProximity'):
                    if self.add_centrality_chars and df_name == 'df_BluetoothProximity':
                        fe = FeatureEngineer(df_for_adv, df_name, advanced=True, add_centrality_chars=True)
                    else:
                        fe = FeatureEngineer(df_for_adv, df_name, advanced=True)
                    df_newname = df_name + '_advanced'
                    self.feature_dfs_forflmat[df_newname] = fe.engineer().drop(['index', 'cnt'], axis=1)
            print "ModelTester: Engineered basic and/or advanced for " + df_name + "\n"


        ''' Merges features and labels into one DataFrame'''
        for feature_df in self.feature_dfs_forflmat.itervalues():
            self.df_labels = self.df_labels.merge(feature_df, how='left', on=['participantID', 'date'])
        self.feature_label_mat = self.df_labels

        self.feature_label_mat = self.feature_label_mat[pd.notnull(self.feature_label_mat['participantID'])]
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

    def create_cv_pipeline(self, n_folds):
        '''
        INPUT: int
        OUTPUT: None

        Divides feature-label matrix into n_folds folds, saving each to, respectively,
        X_train_folds, X_test_folds, y_all_train_folds, and y_all_test_folds.
        Scales features if self.reduce_dimensions set to True.
        To be used in n_folds-fold cross-validation.
        '''

        ''' 1. Pulls out X, y_all (y_all columns include all possible labels) '''
        self.n_folds = n_folds
        n_elems = self.feature_label_mat.shape[0]
        kf = cross_validation.KFold(n_elems, n_folds=n_folds)
        drop_from_X = self.poss_labels + ['participantID', 'date']
        self.features_used = self.feature_label_mat.drop(drop_from_X, axis=1).columns.values
        self.feature_label_mat.sort('participantID', inplace=True)  # Necessary so doesn't "learn" the participants

        if self.reduce_dimensions:
            scaler = StandardScaler()
            self.feature_label_mat[self.features_used] = scaler.fit_transform(self.feature_label_mat[self.features_used])

        X = self.feature_label_mat.drop(drop_from_X, axis=1).values
        y_all = self.feature_label_mat[self.poss_labels].values

        ''' 2. Defines folds and saves to lists'''
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_all_train, y_all_test = y_all[train_index], y_all[test_index]
            self.X_train_folds.append(X_train)
            self.X_test_folds.append(X_test)
            self.y_all_train_folds.append(y_all_train)
            self.y_all_test_folds.append(y_all_test)
        print "Cross-validation folds created"

    def fit_score_models(self, models, energy_kept=0.9):
        '''
        INPUT: dict of model --> string (e.g.,: {rfr: 'Random Forest Regressor', ...})
        OUTPUT: None

        Fits and scores inputted models, printing out k-fold scores and average score.
        Reduces dimensions if self.reduce_dimensions, keeping energy_kept proportion of energy (or # of features).
        Saves feature importances in feature_importances.
        '''

        '''
        INPUT: Float
        OUTPUT: None
        Scales, then uses PCA on self.feature_label_mat to reduce the number of features, preserving
        energy_kept proportion of the original energy (variance).
        '''

        self.models = models    # Mostly to save for future reference
        for model, descrip in models.iteritems():
            mean_scores_by_label, mean_adj_r2_by_label = {}, {}
            for poss_label_col_num, poss_label in enumerate(self.poss_labels):
                scores = np.zeros(self.n_folds)
                for i in xrange(self.n_folds):
                    X_train, X_test = self.X_train_folds[i], self.X_test_folds[i]
                    y_train = self.y_all_train_folds[i][:, poss_label_col_num]

                    # Reducing dimensions: fits on X_train, transforms X_train and X_test
                    if self.reduce_dimensions:
                        pca = PCA(n_components=energy_kept)
                        pca.fit(X_train)
                        X_train = pca.transform(X_train)
                        X_test = pca.transform(X_test)

                    model.fit(X_train, y_train)
                    scores[i] = model.score(X_test, self.y_all_test_folds[i][:, poss_label_col_num])
                print "scores: ", scores
                mean_scores_by_label[poss_label] = np.mean(scores)
                samp_size = self.feature_label_mat.shape[0]
                n_feat = len(self.features_used)

                ''' Feature importances '''
                if not self.reduce_dimensions:
                    importances = np.array(zip(self.features_used, model.feature_importances_))
                    descending_importance_indexes = np.argsort(model.feature_importances_)[::-1]
                    self.feature_importances.append((descrip, poss_label, importances[descending_importance_indexes]))

            ''' R^2, Adjusted R^2 '''
            print "\n\n", descrip
            print "==================================================="
            if self.reduce_dimensions: print "Results with dimensions reduced:"
            for label, score in mean_scores_by_label.iteritems():
                print label, " R^2: ", score
            print "==================================================="
            print "\n"
