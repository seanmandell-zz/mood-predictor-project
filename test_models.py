# from create_labels import create_poss_labels

# CHANGE TO create_labels ####################
from create_labels2 import create_poss_labels
##############################################

from feature_engineer import engineer#_all#, read_in_as_dfs
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn import metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, \
                             GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor


class ModelTester(object):
    '''
    INPUT (class constructor): list of DataFrames
    '''
    def __init__(self, feature_text_files):
        self.feature_dfs = {}
        self.df_labels = create_poss_labels('SurveyFromPhone.csv')
        self.feature_label_mat = None
        self.models = []
        self.X_train_folds, self.X_test_folds, self.y_all_train_folds, self.y_all_test_folds = [], [], [], []
        self.poss_labels, self.n_folds = None, None

        ''' Reads in raw feature_dfs'''
        for text_file in feature_text_files:
            input_name = text_file
            df_name = "df_" + text_file.split('.')[0]
            #globals()[df_name] = pd.read_csv(text_file)
            self.feature_dfs[df_name] = pd.read_csv(text_file)
        print "Labels created, feature dfs read in"

        #self._create_feature_label_mat()
        # for df in self.feature_dfs.itervalues():
        #     feature_cols = list(self.feature_dfs[i].columns)
        #     feature_cols.remove('participantID')
        #     feature_cols.remove('date')
        #     cols_with_some_nan += feature_cols


    def _drop_nan_rows(self):
        '''
        INPUT: None
        OUTPUT: None
        Drops rows in self.feature_label_mat where all values are missing
        '''
        #cols_with_some_nan = []
        for df in self.feature_dfs.itervalues():
            feature_cols = list(df.columns)
            feature_cols.remove('participantID')
            feature_cols.remove('date')
            #cols_with_some_nan += feature_cols
        self.feature_label_mat = self.feature_label_mat[self.feature_label_mat[feature_cols].sum(axis=1) != 0]

    def _fill_na(self):
        '''
        INPUT: None
        OUTPUT: None
        Fills in na values as appropriate
            - Call and SMS: set to 0
            - Battery and BluetoothProximity: set to each participant's median value
                --> For now, battery is just overall median; not enough na values to matter much
        '''

        #CallLog_cols = ['call_incoming', 'call_outgoing', 'call_diff']

        if 'df_CallLog' in self.feature_dfs.keys():
            CallLog_cols = list(self.feature_dfs['df_CallLog'].columns.values)
            CallLog_cols.remove('participantID')
            CallLog_cols.remove('date')
            for col in CallLog_cols:
                self.feature_label_mat.loc[:, col] = self.feature_label_mat[col].fillna(0) # ADDED IN .LOC

        #['sms_incoming', 'sms_outgoing', 'sms_diff']
        if 'df_SMSLog' in self.feature_dfs.keys():
            SMSLog_cols = list(self.feature_dfs['df_SMSLog'].columns.values)
            SMSLog_cols.remove('participantID')
            SMSLog_cols.remove('date')
            for col in SMSLog_cols:
                self.feature_label_mat.loc[:, col] = self.feature_label_mat[col].fillna(0) # ADDED IN .LOC

        #['level', 'plugged', 'temperature', 'voltage']
        if 'df_Battery' in self.feature_dfs.keys():
            Battery_cols = list(self.feature_dfs['df_Battery'].columns.values)
            Battery_cols.remove('participantID')
            Battery_cols.remove('date')
            for col in Battery_cols:
                ''' Now, just using overall median, not by participant '''
                self.feature_label_mat.loc[:, col] = self.feature_label_mat[col].fillna(self.feature_label_mat[col].median()) # ADDED IN .LOC

                # ''' Each participant's median '''
                # df_median_by_partic = pd.DataFrame(self.feature_label_mat.groupby('participantID')[col].median()).reset_index()
                #
                # df_median_by_partic.rename(columns={col: 'median'}, inplace=True)
                # median_series = self.feature_label_mat.merge(df_median_by_partic, how='left', on='participantID')['median']
                # self.feature_label_mat[col] = self.feature_label_mat[col].fillna(median_series)


        BluetoothProximity_cols = ['bt_n', 'bt_n_distinct']
        if 'df_BluetoothProximity' in self.feature_dfs.keys():
            for col in BluetoothProximity_cols:
                # ''' Now, just using overall median, not by participant '''
                # self.feature_label_mat.loc[:, col] = self.feature_label_mat[col].fillna(self.feature_label_mat[col].median()) # ADDED IN .LOC


                ''' Each participant's median '''
                df_median_by_partic = pd.DataFrame(self.feature_label_mat.groupby('participantID')[col].median()).reset_index()

                df_median_by_partic.rename(columns={col: 'median'}, inplace=True)
                median_series = self.feature_label_mat.merge(df_median_by_partic, how='left', on='participantID')['median']
                self.feature_label_mat.loc[:, col] = self.feature_label_mat[col].fillna(median_series)


    def _create_demedianed_cols(self):
        '''
        INPUT: None
        OUTPUT: None
        Creates new "de-medianed" feature columns using each participant's median for each existing
         (Doubles number of feature columns.)
        '''
        all_cols = list(self.feature_label_mat.columns.values)
        cols_to_remove = self.poss_labels + ['participantID', 'date']
        for col in cols_to_remove:
            all_cols.remove(col)
        feature_cols = all_cols

        for col in feature_cols:
            new_col_name = col + "_demedianed"

            ''' Each participant's median '''
            df_median_by_partic = pd.DataFrame(self.feature_label_mat.groupby('participantID')[col].median()).reset_index()
            #df_median_by_partic = pd.DataFrame(df.groupby('participantID')['bt_n'].median()).reset_index()
            df_median_by_partic.rename(columns={col: 'median'}, inplace=True)
            median_series = self.feature_label_mat.merge(df_median_by_partic, how='left', on='participantID')['median']
            self.feature_label_mat[new_col_name] = median_series#self.feature_label_mat[col].fillna(median_series)
            self.feature_label_mat[new_col_name] = self.feature_label_mat[col] - self.feature_label_mat[new_col_name]


    def _add_weekend_col(self, Fri_weekend=True, keep_dow=False):
        '''
        INPUT: bool, bool
        OUTPUT: None
        - Adds a dummy column to feature_label_mat called 'weekend',
        1 if the day of week is Sat/Sun (plus Fri if Fri_weekend=1), 0 otherwise
        - Also keeps day of week if keep_dow is True
        '''
        self.feature_label_mat.loc[:, 'day_of_week'] = self.feature_label_mat['date'].map(lambda x: x.dayofweek)
        day_to_split = 5 - 1 * Fri_weekend
        self.feature_label_mat.loc[self.feature_label_mat['day_of_week'] >= day_to_split, 'weekend'] = 1
        self.feature_label_mat.loc[self.feature_label_mat['day_of_week'] < day_to_split, 'weekend'] = 0

        if not keep_dow:
            self.feature_label_mat.drop('day_of_week', axis=1, inplace=True)



    def create_feature_label_mat(self, poss_labels, day_offset=0, Fri_weekend=True, keep_dow=False):
        '''
        INPUT: list of strings, int, bool, bool
        OUTPUT: None
        Creates a feature-matrix DataFrame, and deals with missing values.
        '''
        self.poss_labels = poss_labels

        ''' Engineers features'''
        for name, feature_df in self.feature_dfs.iteritems():
            self.feature_dfs[name] = engineer(name, feature_df, day_offset)


        ''' Merges features and labels into one DataFrame'''
        for feature_df in self.feature_dfs.itervalues():
            self.df_labels = self.df_labels.merge(feature_df, how='left', on=['participantID', 'date'])
        self.feature_label_mat = self.df_labels


        ''' NEW: drops rows where participantID is null '''
        mt.feature_label_mat = mt.feature_label_mat[pd.notnull(mt.feature_label_mat['participantID'])]


        # ''' Drops rows where all features are NaN '''
        # ''' NO SUCH ROWS'''
        # self._drop_nan_rows()

        ''' Fills in missing values as appropriate '''
        self._fill_na()

        ''' Drops 'cnt nan' column if it exists '''
        if list(self.feature_label_mat.columns).count('cnt nan') > 0:
            self.feature_label_mat.drop('cnt nan', axis=1, inplace=True)

        ''' Creates new columns with differences from each user's median value (for *all* feature columns)'''
        ''' Note (1/11/16): including demedianed doesn't seem to help, and may slightly hurt, AdaBoost '''
        #self._create_demedianed_cols()

        ''' NEW: fills the few remaining null values with 0 '''
        mt.feature_label_mat.fillna(0, inplace=True)


        ''' Adds a dummy 'weekend', 1 for Sat/Sun (and Fri if Fri_weekend=True), 0 otherwise '''
        self._add_weekend_col(Fri_weekend=Fri_weekend, keep_dow=keep_dow)





    def create_cv_pipeline(self, n_folds):
        # Note: Need to build this out myself rather than using, e.g., cross_val_score because
        # I want the same train and test sets for all iterations I'm testing
        ''' 1. Pull out X, y_all (all possible labels contained in y_all) '''
        self.n_folds = n_folds
        n_elems = self.feature_label_mat.shape[0]
        kf = cross_validation.KFold(n_elems, n_folds=n_folds)
        drop_from_X = self.poss_labels + ['participantID', 'date']
        X = self.feature_label_mat.drop(drop_from_X, axis=1).values  # Need to use .values for KFold
        y_all = self.feature_label_mat[self.poss_labels].values   # Need to use .values for KFold

        ''' 2. Define folds and save to lists'''
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
        Fits and scores inputted models, printing out k-fold scores and average score
        '''
        # May be able to make more efficient by not re-creating X_train every time (might not matter much)
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

            print "\n\n", descrip
            print "==================================================="
            for label, score in mean_scores_by_label.iteritems():
                print label, " prediction score (regr-->R^2, classifier-->accur.): ", score

if __name__ == '__main__':
    ''' Reads in files to use as features'''
    feature_text_files = [
                          "SMSLog.csv",
                          "CallLog.csv",
                          "Battery.csv",
                          "BluetoothProximity.csv"
                          ]
    n_folds = 5
    poss_labels = ['happy', 'stressed', 'productive']

    poss_labels += ['happy_dummy', 'very_happy', 'very_unhappy']
    # for label in poss_labels:
    #     dummy_name = label + '_dummy'
    #     very_name = 'very_' + label
    #     very_un_name = 'very_un' + label
    #     poss_labels += [dummy_name, very_name, very_un_name]



    #feature_dfs = [df_SMSLog, df_CallLog], df_Battery]
    mt = ModelTester(feature_text_files)

    mt.create_feature_label_mat(poss_labels, day_offset=0, Fri_weekend=True, keep_dow=True)
    mt.create_cv_pipeline(n_folds)


    ''' Defines models '''
    ''' Regressors '''
    rfr = RandomForestRegressor(n_jobs=-1, random_state=42)
    dtr = DecisionTreeRegressor(max_depth=10)
    abr25 = AdaBoostRegressor(n_estimators=25)
    abr50 = AdaBoostRegressor(n_estimators=50) # Default
    #abr100 = AdaBoostRegressor(n_estimators=100)
    abr50_squareloss = AdaBoostRegressor(n_estimators=50, loss='square')
    abr50_exploss = AdaBoostRegressor(n_estimators=50, loss='exponential')
    gbr = GradientBoostingRegressor()
    gbr_stoch = GradientBoostingRegressor(subsample=0.1) # Default n_estimators (100) much better than 500

    ''' Classifiers '''
    rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
    gbc = GradientBoostingClassifier()


    ''' Loads up model-->description dictionary to pass into fit_score_models '''
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

    #models_all = [rfr, rfc, dtr, abr25, abr50, abr100, abr50_squareloss, abr50_exploss, gbr, gbr_stoch]

    model_descrip_dict = {}
    models_to_use = [
            #   rfr,
            #   dtr,
            #   abr25,
            #   abr50,
            #   abr100,
            #   abr50_squareloss,
            #   abr50_exploss,
              gbr,
            #   gbr_stoch,

              #rfc,
              gbc
            ]
    for model in models_to_use:
        model_descrip_dict[model] = descrips_all[model]

    mt.fit_score_models(model_descrip_dict)


''' FOR IPYTHON -------------------------------------------------------'''
for model, descrip in model_descrip_dict.iteritems():
    mean_scores_by_label = {}
    for poss_label_col_num, poss_label in enumerate(mt.poss_labels):
        scores = np.zeros(mt.n_folds)
        for i in xrange(mt.n_folds):
            X_train = mt.X_train_folds[i]
            y_train = mt.y_all_train_folds[i][:, poss_label_col_num]
            model.fit(X_train, y_train)
            scores[i] = model.score(mt.X_test_folds[i], mt.y_all_test_folds[i][:, poss_label_col_num])
        print "scores: ", scores
        mean_scores_by_label[poss_label] = np.mean(scores)

    for label, score in mean_scores_by_label.iteritems():
        print "\n\n", descrip
        print "==================================================="
        print label, " prediction score (regr-->R^2, classifier-->accur.): ", score
''' END FOR IPYTHON --------------------------------------------------'''

    ''' AdaBoost: 50 (n_est) seems to do slightly (but definitely) better than 100, and maybe better than 25 '''

    # mt.fit_score_models(rfr)
    # #mt.fit_score_models(rfc)
    # mt.fit_score_models(dtr)
    # mt.fit_score_models(abr50)
    # mt.fit_score_models(abr50_squareloss)
    # mt.fit_score_models(abr50_exploss)
    #
    # # print "\n\nFitting, scoring AdaBoost Regressor, 100 estimators: "
    # # mt.fit_score_models(abr100)
    # mt.fit_score_models(abr25)
    # mt.fit_score_models(gbr)
    # mt.fit_score_models(gbr_stoch)




    ''' 1/6/16: NEED TO DO FOR CROSS-VALIDATION TO BE VALID
    2. Before train-test split, sort on date?
    '''


# ''' create_feature_label_mat *************************************** '''
# day_offset=0
# Fri_weekend=True
# keep_dow=True
#
# mt.poss_labels = poss_labels
#
# ''' Engineers features'''
# for name, feature_df in mt.feature_dfs.iteritems():
#     mt.feature_dfs[name] = engineer(name, feature_df, day_offset)
#
# ''' Merges features and labels into one DataFrame'''
# for feature_df in mt.feature_dfs.itervalues():
#     mt.df_labels = mt.df_labels.merge(feature_df, how='left', on=['participantID', 'date'])
# mt.feature_label_mat = mt.df_labels
#
#
# ''' NEW: drops rows where participantID is null '''
# mt.feature_label_mat = mt.feature_label_mat[pd.notnull(mt.feature_label_mat['participantID'])]
#
#
#
#
# ''' Fills in missing values as appropriate '''
# mt._fill_na()
#
# ''' Drops 'cnt nan' column if it exists '''
# if list(mt.feature_label_mat.columns).count('cnt nan') > 0:
#     mt.feature_label_mat.drop('cnt nan', axis=1, inplace=True)
#
#
# ''' Creates new columns with differences from each user's median value (for *all* feature columns)'''
# mt._create_demedianed_cols()
#
#
#
#
# ''' NEW: fills the few remaining null values with 0 '''
# mt.feature_label_mat.fillna(0, inplace=True)
#
# ''' end  create_feature_label_mat *************************************** '''





    # ''' Example; may not want to use this method, though '''
    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')
    #
    #
    # '''
    # 1/6/16 notes on cross-validating/scoring preliminary model
    #     1. Need to iterate over:
    #         (a) 6 possible labels
    #         (b) different scorings
    # '''
    #
    #
    # '''
    # NOTES--Cross-Validating and Keeping Track of All Possible Variations
    # 1. Will create different possible models which are different along:
    #    (a) Model type (random forest, SVM, etc.)
    #    (b) Features (could call 1, 2, 3, in increasing complexity)
    #    (c) Label (happy, calm, stressed, sad, angry, onlypos)
    #
    #    Will want to:
    #    (a) Pickle each (trained) model
    #    (b) Store cross-validated scores
    #    (c) Store sample size trained on?
    #
    # 2. Train-test split: will create same splits to use on every model
    #
    # 3. Grid-search: may eventually want to grid search.
    # '''
