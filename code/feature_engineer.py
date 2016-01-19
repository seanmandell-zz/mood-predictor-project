import pandas as pd
from pandas import Timestamp
from datetime import datetime
from pandas.tseries.offsets import *
import networkx as nx
from networkx.convert_matrix import from_pandas_dataframe

class FeatureEngineer(object):
    def __init__(self, df, df_name, advanced=False, add_centrality_chars=False):
        '''
        INPUT: DataFrame, string, bool, bool
            - df: The DataFrame to engineer.
            - df_name: The name of the DataFrame. Should be df_SMSLog, df_CallLog, df_BluetoothProximity,
                       df_AppRunning, or df_Battery.
            - advanced: Whether to call _daily_stats_most_freq to create the advanced features.
                        Available for df_SMSLog, df_CallLog, df_BluetoothProximity.
            - add_centrality_chars: If advanced=True, whether to add in graph centrality measures
                                    for each participant using Bluetooth data.
        OUTPUT: None

        Class constructor.
        '''
        self.df = df
        self.df_name = df_name
        self.advanced = advanced    # False-->engineer basic features, True-->advanced
        self.add_centrality_chars = add_centrality_chars
        self.init_cols = list(df.columns.values)

        if df_name == 'df_SMSLog':
            self.nickname = 'sms'
            self.target = 'number.hash'

        elif df_name == 'df_CallLog':
            self.nickname = 'call'
            self.df = self.df[self.df['type'] != 'missed']
            self.target = 'number.hash'
            if not advanced:
                self.df['type'] = self.df['type'].map(lambda x: str(x).strip('+'))

        elif df_name == 'df_BluetoothProximity':
            self.nickname = 'bt'
            self.target = 'address'
            if advanced:
                self.target = 'participantID.B'
                self.df = self.df[pd.notnull(self.df['participantID.B'])]
        elif df_name == 'df_Battery':   # No 'target' attribute, because adv. features N/A for battery
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
        weighted Eigenvector centrality (the last being weighted by the df's 'cnt' column).
        '''
        df = df_totals.copy()
        df = df[df['participantID'] > df['participantID.B']]
        G = from_pandas_dataframe(df, 'participantID', 'participantID.B', 'cnt')
        degree_centrality = nx.degree_centrality(G)
        eigen_centrality = nx.eigenvector_centrality(G)
        eigen_centrality_weighted = nx.eigenvector_centrality(G, weight='cnt')

        return degree_centrality, eigen_centrality, eigen_centrality_weighted

    def _totals_for_daily_stats(self):
        '''
        INPUT: None
        OUTPUT: DataFrame

        Helper function called by _daily_stats_most_freq.
        Returns a DataFrame with columns:
            - participantID
            - self.target (e.g., 'participantID.B')
            - 'cnt'
        --> 'cnt' is the total number of interactions (calls, texts, etc.) b/n participantID
            and self.target over the whole dataset.
            In the case of Bluetooth data, which is limited here to interactions between study
            participants, it's the mean number of interactions registered on either party's phone.
        '''
        df_totals = self.df.copy()
        df_totals.loc[:, 'cnt'] = 1
        df_totals = df_totals.groupby(['participantID', self.target])['cnt'].count().reset_index()

        if self.df_name == 'df_BluetoothProximity':
            df_network_cnts2 = df_totals.copy()
            df_totals = df_totals.merge(df_network_cnts2, left_on=['participantID', self.target],\
                                                right_on=[self.target, 'participantID'])
            df_totals['cnt'] = df_totals.mean(axis=1)
            df_totals.rename(columns={'participantID_x': 'participantID', self.target+'_x': self.target}, inplace=True)

        return df_totals[['participantID', self.target, 'cnt']]

    def _perday_for_daily_stats(self, df_totals):
        '''
        INPUT: DataFrame
        OUTPUT: DataFrame

        Helper function called by _daily_stats_most_freq.
        Adds columns to df_totals giving per-day stats for each bucket for every participant.
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
        INPUT: DataFrame
        OUTPUT: None

        Helper function called by _daily_stats_most_freq.
        Adds columns to self.df, giving daily stats for each bucket for every participant.
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

            self.df.loc[self.df['participantID'] == user, self.nickname+'_top1'] = self.df['date'].map(mask1_dict)
            self.df.loc[self.df['participantID'] == user, self.nickname+'_2_4'] = self.df['date'].map(mask_2_4dict)
            self.df.loc[self.df['participantID'] == user, self.nickname+'_5_10'] = self.df['date'].map(mask_5_10dict)
            self.df.loc[self.df['participantID'] == user, self.nickname+'_all'] =  self.df['date'].map(all_dict)
        print "Daily value columns created."


    def _daily_stats_most_freq(self):
        '''
        INPUT: None
        OUTPUT: None

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

        ''' Percent columns '''
        nickname = self.nickname
        self.df.loc[:, nickname+'_top1_pct'] = self.df[nickname+'_top1'].astype(float) / self.df[nickname+'_top1_perday']
        self.df.loc[:, nickname+'_2_4_pct'] = self.df[nickname+'_2_4'].astype(float) / self.df[nickname+'_2_4_perday']
        self.df.loc[:, nickname+'_5_10_pct'] = self.df[nickname+'_5_10'].astype(float) / self.df[nickname+'_5_10_perday']
        self.df.loc[:, nickname+'_all_pct'] = self.df[nickname+'_all'].astype(float) / self.df[nickname+'_all_perday']

        ''' Cleaning up '''
        self.init_cols.remove('participantID')
        self.init_cols.remove('date')
        self.df.drop(self.init_cols, axis=1, inplace=True)
        self.df = self.df.drop_duplicates().reset_index()
        self.df = self.df[pd.notnull(self.df['participantID'])]

        ''' Graph centrality characteristics '''
        if self.add_centrality_chars:
            self.df.loc[:, 'degree_centrality'] = self.df['participantID'].map(degree_centrality)
            self.df.loc[:, 'eigen_centrality'] = self.df['participantID'].map(eigen_centrality)
            self.df.loc[:, 'eigen_centrality_weighted'] = self.df['participantID'].map(eigen_centrality_weighted)

        self.df.fillna(0, inplace=True)
        print self.nickname, "'s daily stats features engineered"

    def engineer_bt(self):
        '''
        INPUT: None
        OUTPUT: None

        Engineers basic features for Bluetooth Proximity DataFrame. Contains columns:
                - participantID
                - date
                - bt_n
                    --> Number of devices a participant is within BT proximity of each day
                - bt_n_distinct
                    --> Number of distinct devices a participant is within BT proximity of each day
        '''

        temp_df_bt_n = self.df.groupby(['participantID', 'date'])['address'].count().reset_index()
        temp_df_bt_n = temp_df_bt_n.rename(columns={'address': 'bt_n'})
        self.df['date'] = self.df['date'].map(lambda x: Timestamp(x))         # Necessary for merge
        temp_df_bt_n['date'] = temp_df_bt_n['date'].map(lambda x: Timestamp(x)) # Necessary for merge
        temp_df_bt_n_distinct = self.df.groupby(['participantID', 'date'])['address'].nunique().reset_index()
        temp_df_bt_n_distinct = temp_df_bt_n_distinct.rename(columns={'address': 'bt_n_distinct'})
        self.df = self.df.merge(temp_df_bt_n, how='left', on=['participantID', 'date'])
        self.df = self.df.merge(temp_df_bt_n_distinct, how='left', on=['participantID', 'date'])
        self.df = self.df[['participantID', 'date', 'bt_n', 'bt_n_distinct']]
        self.df.drop_duplicates(inplace=True)

    def engineer_battery(self):
        '''
        INPUT: None
        OUTPUT: None

        Cleans and engineers battery DataFrame. Contains columns:
            - participantID
            - date
            - level_min, level_mean, level_max
            - plugged_mean
            - temperature_min, temperature_mean, temperature_max
            - voltage_min, voltage_mean, voltage_max
        '''
        self.df.loc[self.df['plugged'] > 1, 'plugged'] = 1
        df_new = self.df[['participantID', 'date']].drop_duplicates().reset_index().drop('index', axis=1)
        min_mean_max_cols = ['level', 'plugged', 'temperature', 'voltage']
        for col in min_mean_max_cols:
            min_name = col + "_min"
            mean_name = col + "_mean"
            max_name = col + "_max"
            grouped = self.df.groupby(['participantID', 'date'])[col]
            df_new[min_name] = grouped.min().reset_index()[col]
            df_new[mean_name] = grouped.mean().reset_index()[col]
            df_new[max_name] = grouped.max().reset_index()[col]
        df_new.drop(['plugged_min', 'plugged_max'], axis=1, inplace=True)

        self.df = df_new

    def engineer(self):
        '''
        INPUT: None
        OUTPUT: DataFrame

        Engineers a raw DataFrame and returns it.
        '''
        if (self.df_name == 'df_SMSLog' or self.df_name == 'df_CallLog'):
            self.df.rename(columns={'participantID.A': 'participantID'}, inplace=True)
            self.init_cols.remove('participantID.A')
            self.init_cols.append('participantID')
            if not self.advanced:
                self._calc_incoming_outgoing()
            else:
                self._daily_stats_most_freq()

        if self.df_name == 'df_BluetoothProximity':
            if not self.advanced:
                self.engineer_bt()
            else:
                self._daily_stats_most_freq()

        if self.df_name == 'df_Battery':
            self.engineer_battery()

        ''' Converts 'date' column to Timestamp if necessary (so merge with df_labels works)'''
        if self.df['date'][0].__class__.__name__ != 'Timestamp':
            self.df['date'] = self.df['date'].map(lambda x: Timestamp(x))

        if not self.advanced:
            print self.df_name + " basic features engineered"
        else:
            print self.df_name + " advanced features engineered"
        return self.df
