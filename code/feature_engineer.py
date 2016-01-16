import pandas as pd
from pandas import Timestamp
from datetime import datetime
from pandas.tseries.offsets import *
import networkx as nx
from networkx.convert_matrix import from_pandas_dataframe

def _calc_incoming_outgoing(df):
    '''
    INPUT: DataFrame with columns 'participantID.A', 'type', 'date' (and others)
    OUTPUT: DataFrame with columns participantID, date, sms_incoming, sms_outgoing, sms_diff
                --> Need to rename if don't want 'sms' in resulting columns

    Calculates counts of incoming and outgoing texts each day for each participant.
    Called by engineer_sms and engineer_call.
    '''
    df['cnt'] = 1
    df = pd.DataFrame(df.groupby(['participantID.A', 'type', 'date'])['cnt'].count()).reset_index()
    df = df.set_index(['participantID.A', 'date', 'type'])
    df = df.unstack()
    df = df.reset_index()
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    df = df.fillna(0)
    df = df.rename(columns={'participantID.A': 'participantID', 'cnt incoming': 'sms_incoming', 'cnt outgoing': 'sms_outgoing'})
    df['sms_diff'] = df['sms_incoming'] - df['sms_outgoing']
    df['sms_total'] = df['sms_incoming'] + df['sms_outgoing']
    return df

def _graph_centrality_measures(df):
    '''
    INPUT: DataFrame
    OUTPUT: dict, dict, dict

    For every participant, calculates degree centrality, Eigenvector centrality, and
    weighted Eigenvector centrality (weighted by the df's 'mean_cnt' column).
    '''
    df = df[df['participantID.A'] > df['participantID.B']]
    G = from_pandas_dataframe(df, 'participantID.A', 'participantID.B', 'mean_cnt')

    degree_centrality = nx.degree_centrality(G)
    eigen_centrality = nx.eigenvector_centrality(G)
    eigen_centrality_weighted = nx.eigenvector_centrality(G, weight='mean_cnt')

    return degree_centrality, eigen_centrality, eigen_centrality_weighted

'''
TO GENERALIZE
Need to make calculating the mean of both directions optional
'''
def _totals_for_daily_stats(df, partic_name, target_name):
    '''
    Input: cleaned (date-limited, etc.) df
    Returns df further cleaned:
    Note: calculates mean
    '''
    df_network = df.copy()
    df_network.loc[:, 'cnt'] = 1
    df_network_cnts = df_network.groupby([partic_name, target_name])['cnt'].count().reset_index()

    ''' NEED TO CHANGE: pass in parameter like bidirectional; if bidirectional:'''
    if True:
        df_network_cnts2 = df_network_cnts.copy()
        df_network_merged = df_network_cnts.merge(df_network_cnts2, left_on=[partic_name, target_name],\
                                            right_on=[target_name, partic_name])
        df_network_merged['mean_cnt'] = df_network_merged.mean(axis=1)
        df_network_merged.rename(columns={partic_name+'_x': 'participantID.A', target_name+'_x': 'participantID.B'}, inplace=True)

    df_network_merged = df_network_merged[['participantID.A', 'participantID.B', 'mean_cnt']]

    return df_network_merged



def _perday_for_daily_stats(df, df_totals, nickname):
    '''
    INPUT: DataFrame, DataFrame, string
    OUTPUT: DataFrame

    Adds columns to df_totals giving per-day stats for each bucket for every participant.
    Called by _daily_stats_most_freq.
    '''

    ''' Creates [nickname]_top1, [nickname]_2_4, [nickname]_5_10, [nickname]_all'''
    for user in df_totals['participantID.A'].unique():
        df_totals.loc[df_totals['participantID.A'] == user, nickname+'_top1'] = \
                    sum(df_totals[df_totals['participantID.A'] == user].iloc[:1]['mean_cnt'])
        df_totals.loc[df_totals['participantID.A'] == user, nickname+'_2_4'] = \
                    sum(df_totals[df_totals['participantID.A'] == user].iloc[1:4]['mean_cnt'])
        df_totals.loc[df_totals['participantID.A'] == user, nickname+'_5_10'] = \
                    sum(df_totals[df_totals['participantID.A'] == user].iloc[4:10]['mean_cnt'])
        df_totals.loc[df_totals['participantID.A'] == user, nickname+'_all'] = \
                    sum(df_totals[df_totals['participantID.A'] == user]['mean_cnt'])

    ''' Creates the above but normalized to a per-day basis '''
    df_totals['n_days_partic'] = df_totals['participantID.A'].map(dict(df.groupby('participantID')['date'].nunique()))
    df_totals.loc[:, nickname+'_top1_perday'] = df_totals[nickname+'_top1'].astype(float) / df_totals['n_days_partic']
    df_totals.loc[:, nickname+'_2_4_perday'] = df_totals[nickname+'_2_4'].astype(float) / df_totals['n_days_partic']
    df_totals.loc[:, nickname+'_5_10_perday'] = df_totals[nickname+'_5_10'].astype(float) / df_totals['n_days_partic']
    df_totals.loc[:, nickname+'_all_perday'] = df_totals[nickname+'_all'].astype(float) / df_totals['n_days_partic']
    cols_to_drop = [nickname+'_top1', nickname+'_2_4', nickname+'_5_10', nickname+'_all', 'n_days_partic']
    df_totals.drop(cols_to_drop, axis=1, inplace=True)
    print nickname, "daily stats per-day columns created. Creating daily value columns..."

    ''' Per-day and percent columns--modifying df '''
    perday_cols = [nickname+'_top1_perday', nickname+'_2_4_perday', nickname+'_5_10_perday', nickname+'_all_perday']
    dnm_collapsed = df_totals[perday_cols + ['participantID.A']].drop_duplicates()
    for col in perday_cols:
        col_dict = dict(dnm_collapsed[['participantID.A', col]].set_index('participantID.A')[col])
        df[col] = df['participantID'].map(col_dict)

    return df, df_totals

def _daily_for_daily_stats(df, df_totals, nickname):
    '''
    INPUT: DataFrame, DataFrame, string
    OUTPUT: DataFrame

    Adds columns to df, giving daily stats for each bucket for every participant.
    Called by _daily_stats_most_freq.
    '''
    df.loc[:, 'cnt'] = 1
    for user in df_totals['participantID.A'].unique():
        top10 = list(df_totals[df_totals['participantID.A'] == user].iloc[:10]['participantID.B'])
        top1 = top10[:1]
        top_2_4 = top10[1:4]
        top_5_10 = top10[4:10]
        df_temp = df[df['participantID'] == user]
        mask1 = df_temp['participantID.B'] == top1[0]
        mask_2_4 = df_temp['participantID.B'].map(lambda x: top_2_4.count(x) > 0)
        mask_5_10 = df_temp['participantID.B'].map(lambda x: top_5_10.count(x) > 0)

        mask1_dict = dict(df_temp[mask1].groupby('date')['cnt'].count())
        mask_2_4dict = dict(df_temp[mask_2_4].groupby('date')['cnt'].count())
        mask_5_10dict = dict(df_temp[mask_5_10].groupby('date')['cnt'].count())
        all_dict = dict(df_temp.groupby('date')['cnt'].count())

        df.loc[df['participantID'] == user, nickname+'_top1'] = df['date'].map(mask1_dict)
        df.loc[df['participantID'] == user, nickname+'_2_4'] = df['date'].map(mask_2_4dict)
        df.loc[df['participantID'] == user, nickname+'_5_10'] = df['date'].map(mask_5_10dict)
        df.loc[df['participantID'] == user, nickname+'_all'] =  df['date'].map(all_dict)
    print "Daily value columns created."

    return df


''' WILL TAKE OUT DEFAULT VALUES FOR ARGUMENTS '''
def _daily_stats_most_freq(df, nickname, partic_name='participantID', target_name='participantID.B', add_centrality_chars=False):
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
    df_totals = _totals_for_daily_stats(df, partic_name, target_name).sort(['participantID.A', 'mean_cnt'], ascending=False)
    if add_centrality_chars:
        degree_centrality, eigen_centrality, eigen_centrality_weighted = _graph_centrality_measures(df_totals)
    df, df_totals = _perday_for_daily_stats(df, df_totals, nickname)
    df = _daily_for_daily_stats(df, df_totals, nickname)

    ''' Percent columns '''
    df[nickname+'_top1_pct'] = df[nickname+'_top1'].astype(float) / df[nickname+'_top1_perday']
    df[nickname+'_2_4_pct'] = df[nickname+'_2_4'].astype(float) / df[nickname+'_2_4_perday']
    df[nickname+'_5_10_pct'] = df[nickname+'_5_10'].astype(float) / df[nickname+'_5_10_perday']
    df[nickname+'_all_pct'] = df[nickname+'_all'].astype(float) / df[nickname+'_all_perday']

    ''' Cleaning up '''
    df.drop(['participantID.B', 'address', 'cnt'], axis=1, inplace=True)
    df = df.drop_duplicates().reset_index()
    df = df[pd.notnull(df['participantID'])]

    ''' Graph centrality characteristics '''
    if add_centrality_chars:
        df.loc[:, 'degree_centrality'] = df['participantID'].map(degree_centrality)
        df.loc[:, 'eigen_centrality'] = df['participantID'].map(eigen_centrality)
        df.loc[:, 'eigen_centrality_weighted'] = df['participantID'].map(eigen_centrality_weighted)

    df.fillna(0, inplace=True)
    print nickname, "'s daily stats features engineered"
    return df




''' INCOMPLETE FUNCTION; MAY DELETE'''
def engineer_app(df):
    '''
    INPUT: DataFrame with raw App Running data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
    '''

    df['app'] = df['package'].map(lambda x: x.split('.')[-1])
    df = _daily_stats_most_freq(df, nickname='app', partic_name='participantID', target_name='app')




def engineer_bt(df):
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

def engineer_sms(df):
    '''
    INPUT: DataFrame with raw SMS log data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
            - participantID
            - date
            - sms_incoming (count)
            - sms_outgoing (count)
            - sms_diff (count equaling (sms_incoming-sms_outgoing))
    Calls _calc_incoming_outgoing, which calculates counts of incoming and outgoing texts each day for each participant
    '''
    df = _calc_incoming_outgoing(df)
    return df

def engineer_call(df):
    '''
    INPUT: DataFrame with raw call log data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
            - participantID
            - date
            - call_incoming (count)
            - call_outgoing (count)
            - call_diff (count equaling (call_incoming-call_outgoing))
    '''
    ''' Drops missed calls and strips + from outgoing+ and incoming+ '''
    df = df[df['type'] != 'missed']
    df['type'] = df['type'].map(lambda x: str(x).strip('+'))
    df = _calc_incoming_outgoing(df)
    df = df.rename(columns={'sms_diff': 'call_diff', 'sms_incoming': 'call_incoming', \
                            'sms_outgoing': 'call_outgoing', 'sms_total': 'call_total'})
    return df

def engineer_battery(df):
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

def engineer(name, feature_df, basic_call_sms_bt_features, advanced_call_sms_bt_features, other_features):
    '''
    INPUT: string, DataFrame
    OUTPUT: DataFrame
    Engineers a raw DataFrame and returns it.
    --> name is the name of the raw DataFrame
    'weekend' is a dummy var equal to 1 for Friday, Saturday, and Sunday
    '''


    if basic_call_sms_bt_features:
        if name == 'df_SMSLog':
            feature_df = engineer_sms(feature_df)
            print "df_SMSLog basic features engineered"
        elif name == 'df_CallLog':
            feature_df = engineer_call(feature_df)
            print "df_CallLog basic features engineered"
        elif name == 'df_Battery':
            feature_df = engineer_battery(feature_df)
            print "df_Battery basic features engineered"

    if name == 'df_BluetoothProximity':
        feature_df = engineer_bt(feature_df)

        print "df_BluetoothProximity engineered"


    ''' Converts 'date' column to Timestamp if necessary (so merge with df_labels works)'''
    if feature_df['date'][0].__class__.__name__ != 'Timestamp':
        feature_df['date'] = feature_df['date'].map(lambda x: Timestamp(x))


    return feature_df


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
