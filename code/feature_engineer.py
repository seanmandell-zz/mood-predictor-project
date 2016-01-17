import pandas as pd
from pandas import Timestamp
from datetime import datetime
from pandas.tseries.offsets import *
import networkx as nx
from networkx.convert_matrix import from_pandas_dataframe

class FeatureEngineer(object):
    def __init__(self, df, df_name, advanced=False, add_centrality_chars=False):#, target, ):


        self.df = df
        self.df_name = df_name
        self.advanced = advanced    # 0=basic, 1=advanced
        self.add_centrality_chars = add_centrality_chars

        if df_name == 'df_SMSLog':
            #code
            self.nickname = 'sms'
            self.target = 'number.hash'

        elif df_name == 'df_CallLog':
            #code
            self.nickname = 'call'
            self.df = self.df[self.df['type'] != 'missed']
            self.target = 'number.hash'
            if not advanced:
                self.df['type'] = self.df['type'].map(lambda x: str(x).strip('+'))

        elif df_name == 'df_BluetoothProximity':
            #code
            self.nickname = 'bt'
            self.target = 'address'
            if advanced:
                self.target = 'participantID.B'
                self.df = self.df[pd.notnull(self.df['participantID.B'])]
        elif df_name == 'df_AppRunning':
            #code
            self.nickname = 'app'
        elif df_name == 'df_Battery':
            #code
            self.nickname = 'battery'


    def _calc_incoming_outgoing(self):
        '''
        INPUT: None
        OUTPUT: None

        Calculates counts of incoming and outgoing texts/calls each day for each participant.
        Results in columns: participantID, date, [nickname]_incoming, [nickname]_outgoing, [nickname]_diff
        '''
        self.df['cnt'] = 1
        self.df = pd.DataFrame(self.df.groupby(['participantID', 'type', 'date'])['cnt'].count()).reset_index()
        self.df = self.df.set_index(['participantID', 'date', 'type'])
        self.df = self.df.unstack()
        self.df = self.df.reset_index()
        self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        self.df = self.df.fillna(0)
        self.df = self.df.rename(columns={'cnt incoming': self.nickname+'_incoming', 'cnt outgoing': self.nickname+'_outgoing'})
        self.df[self.nickname+'_diff'] = self.df[self.nickname+'_incoming'] - self.df[self.nickname+'_outgoing']
        self.df[self.nickname+'_total'] = self.df[self.nickname+'_incoming'] + self.df[self.nickname+'_outgoing']

    def _graph_centrality_measures(self, df_totals):
        '''
        INPUT: DataFrame
        OUTPUT: dict, dict, dict

        For every participant, calculates degree centrality, Eigenvector centrality, and
        weighted Eigenvector centrality (weighted by the df's 'cnt' column).
        '''
        df = df_totals.copy()
        df = df[df['participantID'] > df['participantID.B']]
        G = from_pandas_dataframe(df, 'participantID', 'participantID.B', 'cnt')

        degree_centrality = nx.degree_centrality(G)
        eigen_centrality = nx.eigenvector_centrality(G)
        eigen_centrality_weighted = nx.eigenvector_centrality(G, weight='cnt')

        return degree_centrality, eigen_centrality, eigen_centrality_weighted

    '''
    TO GENERALIZE
    Need to make calculating the mean of both directions optional
    '''
    def _totals_for_daily_stats(self):
        '''
        Input: cleaned (date-limited, etc.) df
        Returns df further cleaned:
        Note: calculates mean
        '''
        self.df.loc[:, 'cnt'] = 1

        # print "self.df.head(): \n", self.df.head(), "\n"
        # print "partic_name: ", partic_name
        # print "target_name: ", target_name

        self.df = self.df.groupby(['participantID', self.target])['cnt'].count().reset_index()

        if self.df_name == 'df_BluetoothProximity':
            df_network_cnts2 = self.df.copy()
            self.df = self.df.merge(df_network_cnts2, left_on=['participantID', self.target],\
                                                right_on=[self.target, 'participantID'])
            self.df['cnt'] = self.df.mean(axis=1)
            self.df.rename(columns={'participantID_x': 'participantID', self.target+'_x': self.target}, inplace=True)

        self.df = self.df[['participantID', self.target, 'cnt']]


    def _perday_for_daily_stats(self, df_totals):
        '''
        INPUT: DataFrame, DataFrame, string
        OUTPUT: DataFrame

        Adds columns to df_totals giving per-day stats for each bucket for every participant.
        Called by _daily_stats_most_freq.
        '''

        nickname = self.nickname
        ''' Creates [nickname]_top1, [nickname]_2_4, [nickname]_5_10, [nickname]_all'''
        for user in df_totals['participantID'].unique():
            df_totals.loc[df_totals['participantID'] == user, nickname+'_top1'] = \
                        sum(df_totals[df_totals['participantID'] == user].iloc[:1]['cnt'])
            df_totals.loc[df_totals['participantID'] == user, nickname+'_2_4'] = \
                        sum(df_totals[df_totals['participantID'] == user].iloc[1:4]['cnt'])
            df_totals.loc[df_totals['participantID'] == user, nickname+'_5_10'] = \
                        sum(df_totals[df_totals['participantID'] == user].iloc[4:10]['cnt'])
            df_totals.loc[df_totals['participantID'] == user, nickname+'_all'] = \
                        sum(df_totals[df_totals['participantID'] == user]['cnt'])

        ''' Creates the above but normalized to a per-day basis '''
        df_totals['n_days_partic'] = df_totals['participantID'].map(dict(self.df.groupby('participantID')['date'].nunique()))
        df_totals.loc[:, nickname+'_top1_perday'] = df_totals[nickname+'_top1'].astype(float) / df_totals['n_days_partic']
        df_totals.loc[:, nickname+'_2_4_perday'] = df_totals[nickname+'_2_4'].astype(float) / df_totals['n_days_partic']
        df_totals.loc[:, nickname+'_5_10_perday'] = df_totals[nickname+'_5_10'].astype(float) / df_totals['n_days_partic']
        df_totals.loc[:, nickname+'_all_perday'] = df_totals[nickname+'_all'].astype(float) / df_totals['n_days_partic']
        cols_to_drop = [nickname+'_top1', nickname+'_2_4', nickname+'_5_10', nickname+'_all', 'n_days_partic']
        df_totals.drop(cols_to_drop, axis=1, inplace=True)
        print nickname, "daily stats per-day columns created. Creating daily value columns..."

        ''' Per-day and percent columns--modifying df '''
        perday_cols = [nickname+'_top1_perday', nickname+'_2_4_perday', nickname+'_5_10_perday', nickname+'_all_perday']
        dnm_collapsed = df_totals[perday_cols + ['participantID']].drop_duplicates()
        for col in perday_cols:
            col_dict = dict(dnm_collapsed[['participantID', col]].set_index('participantID')[col])
            self.df[col] = self.df['participantID'].map(col_dict)

        return df_totals

    def _daily_for_daily_stats(self, df_totals):
        '''
        INPUT: DataFrame, DataFrame, string
        OUTPUT: DataFrame

        Adds columns to df, giving daily stats for each bucket for every participant.
        Called by _daily_stats_most_freq.
        '''
        self.df.loc[:, 'cnt'] = 1
        for user in df_totals['participantID'].unique():
            top10 = list(df_totals[df_totals['participantID'] == user].iloc[:10][self.target])
            top1 = top10[:1]
            top_2_4 = top10[1:4]
            top_5_10 = top10[4:10]
            df_temp = self.df[self.df['participantID'] == user]
            mask1 = df_temp[self.target] == top1[0]
            mask_2_4 = df_temp[self.target].map(lambda x: top_2_4.count(x) > 0)
            mask_5_10 = df_temp[self.target].map(lambda x: top_5_10.count(x) > 0)

            mask1_dict = dict(df_temp[mask1].groupby('date')['cnt'].count())
            mask_2_4dict = dict(df_temp[mask_2_4].groupby('date')['cnt'].count())
            mask_5_10dict = dict(df_temp[mask_5_10].groupby('date')['cnt'].count())
            all_dict = dict(df_temp.groupby('date')['cnt'].count())

            self.df.loc[self.df['participantID'] == user, nickname+'_top1'] = self.df['date'].map(mask1_dict)
            self.df.loc[self.df['participantID'] == user, nickname+'_2_4'] = self.df['date'].map(mask_2_4dict)
            self.df.loc[self.df['participantID'] == user, nickname+'_5_10'] = self.df['date'].map(mask_5_10dict)
            self.df.loc[self.df['participantID'] == user, nickname+'_all'] =  self.df['date'].map(all_dict)
        print "Daily value columns created."


    ''' WILL TAKE OUT DEFAULT VALUES FOR ARGUMENTS '''
    def _daily_stats_most_freq(self):
        '''
        INPUT: DataFrame, string, string, string, bool
            - nickname: string to prepend to new column names
            - partic_name: name of participant column in df
            - target_name: name of column the count of interactions with which are being tallied
            - add_centrality_chars: whether to add 3 columns for measures of graph centrality
        OUTPUT: DataFrame

        For each participant in df, adds columns related to interactions with target_name
        (which is likely another person but could also be, e.g., an app).
        The sets of columns are:
            (a) mean interactions per-day,
            (b) actual daily interactions, and
            (c) a percent calculated as (b) / (a)
        If add_centrality_chars is set to True (should only be so for Bluetooth Proximity),
        3 more columns with 3 centrality figures (over the whole dataset time period, not daily)
        are added.
        '''
        df_totals = self._totals_for_daily_stats()
        df_totals.sort(['participantID', 'cnt'], ascending=False, inplace=True)
        if self.add_centrality_chars:
            degree_centrality, eigen_centrality, eigen_centrality_weighted = self._graph_centrality_measures(df_totals)
        df_totals = self._perday_for_daily_stats(df_totals)
        self._daily_for_daily_stats(df_totals)

        nickname = self.nickname
        ''' Percent columns '''
        self.df[nickname+'_top1_pct'] = self.df[nickname+'_top1'].astype(float) / self.df[nickname+'_top1_perday']
        self.df[nickname+'_2_4_pct'] = self.df[nickname+'_2_4'].astype(float) / self.df[nickname+'_2_4_perday']
        self.df[nickname+'_5_10_pct'] = self.df[nickname+'_5_10'].astype(float) / self.df[nickname+'_5_10_perday']
        self.df[nickname+'_all_pct'] = self.df[nickname+'_all'].astype(float) / self.df[nickname+'_all_perday']

        ''' Cleaning up '''
        self.df.drop(['participantID.B', 'address', 'cnt'], axis=1, inplace=True)
        self.df = self.df.drop_duplicates().reset_index()
        self.df = self.df[pd.notnull(df['participantID'])]

        ''' Graph centrality characteristics '''
        if self.add_centrality_chars:
            self.df.loc[:, 'degree_centrality'] = self.df['participantID'].map(degree_centrality)
            self.df.loc[:, 'eigen_centrality'] = self.df['participantID'].map(eigen_centrality)
            self.df.loc[:, 'eigen_centrality_weighted'] = self.df['participantID'].map(eigen_centrality_weighted)

        self.df.fillna(0, inplace=True)
        print self.nickname, "'s daily stats features engineered"


    ''' INCOMPLETE FUNCTION; MAY DELETE'''
    def engineer_app(self, df):
        '''
        INPUT: DataFrame with raw App Running data
        OUTPUT: DataFrame--cleaned and engineered. Contains columns:
        '''

        df['app'] = df['package'].map(lambda x: x.split('.')[-1])
        df = _daily_stats_most_freq(df, bidirectional=False, nickname='app', partic_name='participantID', target_name='app')




    def engineer_bt(self, df):
        '''
        INPUT: DataFrame with raw Bluetooth Proximity data
        OUTPUT: DataFrame--cleaned and engineered. Contains columns:
                - participantID
                - date
                - bt_n
                    --> Number of devices a participant is within BT proximity of each day
                - bt_n_distinct
                    --> Number of distinct devices a participant is within BT proximity of each day
        '''

        ''' Limits dates to relevant period; removes possibly erroneous nighttime observations'''
        # df = df.rename(columns={'date': 'local_time'})
        # df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
        # ''' Per Friends and Family paper (8.2.1), removes b/n midnight and 7 AM '''
        # df = df[df['local_time'].dt.hour >= 7]
        # df = _limit_dates(df)

        temp_df_bt_n = df.groupby(['participantID', 'date'])['address'].count().reset_index()
        temp_df_bt_n = temp_df_bt_n.rename(columns={'address': 'bt_n'})
        df['date'] = df['date'].map(lambda x: Timestamp(x))         # Necessary for merge
        temp_df_bt_n['date'] = temp_df_bt_n['date'].map(lambda x: Timestamp(x)) # Necessary for merge
        temp_df_bt_n_distinct = df.groupby(['participantID', 'date'])['address'].nunique().reset_index()
        temp_df_bt_n_distinct = temp_df_bt_n_distinct.rename(columns={'address': 'bt_n_distinct'})
        df = df.merge(temp_df_bt_n, how='left', on=['participantID', 'date'])
        df = df.merge(temp_df_bt_n_distinct, how='left', on=['participantID', 'date'])
        df = df[['participantID', 'date', 'bt_n', 'bt_n_distinct']]
        df.drop_duplicates(inplace=True)

        return df

    # def engineer_sms(self):
    #     '''
    #     INPUT: DataFrame with raw SMS log data
    #     OUTPUT: DataFrame--cleaned and engineered. Contains columns:
    #             - participantID
    #             - date
    #             - sms_incoming (count)
    #             - sms_outgoing (count)
    #             - sms_diff (count equaling (sms_incoming-sms_outgoing))
    #     Calls _calc_incoming_outgoing, which calculates counts of incoming and outgoing texts each day for each participant
    #     '''
    #     self._calc_incoming_outgoing()
    #
    # def engineer_call(self, df):
    #     '''
    #     INPUT: DataFrame with raw call log data
    #     OUTPUT: DataFrame--cleaned and engineered. Contains columns:
    #             - participantID
    #             - date
    #             - call_incoming (count)
    #             - call_outgoing (count)
    #             - call_diff (count equaling (call_incoming-call_outgoing))
    #     '''
    #     ''' Drops missed calls and strips + from outgoing+ and incoming+ '''
    #     # df = df[df['type'] != 'missed']
    #     # df['type'] = df['type'].map(lambda x: str(x).strip('+'))
    #     self._calc_incoming_outgoing()


    def engineer_battery(self, df):
        '''
        INPUT: DataFrame with raw battery data
        OUTPUT: DataFrame

        Cleans and engineers battery DataFrame. Contains columns:
            - participantID
            - date
            - level_min, level_mean, level_max
            - plugged_mean
            - temperature_min, temperature_mean, temperature_max
            - voltage_min, voltage_mean, voltage_max
        '''
        df.loc[df['plugged'] > 1, 'plugged'] = 1
        df_new = df[['participantID', 'date']].drop_duplicates().reset_index().drop('index', axis=1)
        min_mean_max_cols = ['level', 'plugged', 'temperature', 'voltage']
        for col in min_mean_max_cols:
            min_name = col + "_min"
            mean_name = col + "_mean"
            max_name = col + "_max"
            grouped = df.groupby(['participantID', 'date'])[col]
            df_new[min_name] = grouped.min().reset_index()[col]
            df_new[mean_name] = grouped.mean().reset_index()[col]
            df_new[max_name] = grouped.max().reset_index()[col]
        df_new.drop(['plugged_min', 'plugged_max'], axis=1, inplace=True)

        return df_new

    def engineer(self):
        '''
        INPUT: None
        OUTPUT: DataFrame
        Engineers a raw DataFrame and returns it.
        --> name is the name of the raw DataFrame
        'weekend' is a dummy var equal to 1 for Friday, Saturday, and Sunday
        '''

        if self.df_name == 'df_SMSLog' or self.df_name == 'df_CallLog':
            self.df.rename(columns={'participantID.A': 'participantID'})
            if not self.advanced:
                self._calc_incoming_outgoing()
            else:
                self._daily_stats_most_freq()






        if basic_call_sms_bt_features:
            if name == 'df_SMSLog':
                feature_df = engineer_sms(feature_df)
            elif name == 'df_CallLog':
                feature_df = engineer_call(feature_df)
            elif name == 'df_Battery':
                feature_df = engineer_battery(feature_df)

        if name == 'df_BluetoothProximity':
            feature_df = engineer_bt(feature_df)








        ''' Converts 'date' column to Timestamp if necessary (so merge with df_labels works)'''
        if self.df['date'][0].__class__.__name__ != 'Timestamp':
            self.df['date'] = self.df['date'].map(lambda x: Timestamp(x))

        if not self.advanced:
            print self.df_name + " basic features engineered"
        else:
            print self.df_name + " advanced features engineered"
        return self.df


    # def engineer_all(feature_dfs):
    #     '''
    #     INPUT: dict (k:v --> name:DataFrame)
    #     OUTPUT: 3 DataFrames, engineered
    #     '''
    #     # engineered_feature_dfs = []
    #     # for feature_df in feature_dfs:
    #     #     engineered_feature_dfs.append(engineer(feature_df))
    #     # return engineered_feature_dfs
    #     #
    #     # ''' HERE, WANT TO ITERATE THROUGH LIST OF FEAT_DFS AND CALL engineer(feat_df),
    #     # which itself will be a function calling the appropriate engineer function
    #     # '''
    #
    #     df_SMSLog = engineer_sms(df_SMSLog)
    #     print "df_SMSLog engineered"
    #     df_CallLog = engineer_call(df_CallLog)
    #     print "df_CallLog engineered"
    #     df_Battery = engineer_battery(df_Battery)
    #     print "df_Battery engineered"
    #     return df_SMSLog, df_CallLog, df_Battery
