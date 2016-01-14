import pandas as pd
from pandas import Timestamp
from datetime import datetime
from pandas.tseries.offsets import *

# def _limit_dates(df, min_date='2010-03-15', max_date='2010-09-05'):
def _limit_dates(df, min_date='2010-11-12', max_date='2011-05-21', day_offset=0):
    '''
    INPUT: DataFrame with local_time column, string, string
    OUTPUT: DataFrame, with local_time column replaced by date
    Helper function called by engineer_... functions.
    Keeps observations within [min_date, max_date], inclusive (where a day is defined as 4 AM to 4 AM the next day).
    If day_offset is positive, features are used to predict moods day_offset days in the future.
    '''

    df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
    df.loc[df['local_time'].dt.hour < 4, 'local_time'] = (pd.DatetimeIndex(df[df['local_time'].dt.hour < 4]['local_time']) - \
                                                         DateOffset(1))


    df['date'] = df['local_time'].dt.date
    df = df.drop('local_time', axis=1)

    if day_offset > 0:
        timedelta_string = str(day_offset) + ' days'
        df.loc[:, 'date'] = df['date'].map(lambda x: x + pd.to_timedelta(timedelta_string))  # Want to add, not subtract

    df = df[((df['date'] >= datetime.date(pd.to_datetime(min_date))) & \
             (df['date'] <= datetime.date(pd.to_datetime(max_date))))]
    return df

def _calc_incoming_outgoing(df):
    '''
    INPUT: DataFrame with columns 'participantID.A', 'type', 'date' (and others)
    OUTPUT: DataFrame with columns participantID, date, sms_incoming, sms_outgoing, sms_diff
                --> Need to rename if don't want SMS
    Helper function called by engineer_sms and engineer_call.
    Calculates counts of incoming and outgoing texts each day for each participant.
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

def _daily_stats_most_popular(df, partic_name='participantID', target_name='app'):
    '''
    INPUT:
    OUTPUT:

    '''
    pass

# def _date_range_by_participant(df, date_min_name, date_max_name):
#     '''
#     INPUT: DataFrame, string, string
#     OUTPUT: DataFrame
#     Creates two new columns for df:
#         - Earliest date a participant shows up in the df
#         - Latest date a participant shows up in the df
#     Used in test_models to fillna appropriately depending on whether the missing data occurs
#     within the user's date range
#     '''
#     min_df = pd.DataFrame(df.groupby('participantID')['date'].min()).rename(columns={'date': date_min_name}).reset_index()
#     max_df = pd.DataFrame(df.groupby('participantID')['date'].max()).rename(columns={'date': date_max_name}).reset_index()
#     df = df.merge(min_df, on='participantID')
#     df = df.merge(max_df, on='participantID')
#     return df
#
#     # min_df = pd.DataFrame(df_SMSLog.groupby('participantID')['date'].min()).rename(columns={'date': 'date_min_sms'})
#     # df_SMSLog = df_SMSLog.merge(min_df, on='participantID')



def engineer_app(df, day_offset):
    '''
    INPUT: DataFrame with raw App Running data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
    '''
    ''' Limits dates to relevant period'''
    df = df.rename(columns={'scantime': 'local_time'})
    df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
    df = _limit_dates(df, day_offset=day_offset)

    ''' Engineers '''
    df['app'] = df['package'].map(lambda x: x.split('.')[-1])
    df = _daily_stats_most_popular(df, partic_name='participantID', target_name='app')



def engineer_bt(df, day_offset):
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
    df = df.rename(columns={'date': 'local_time'})
    df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
    ''' Per Friends and Family paper (8.2.1), removes b/n midnight and 7 AM '''
    df = df[df['local_time'].dt.hour >= 7]
    df = _limit_dates(df, day_offset=day_offset)


    ''' Creates bt_n and bt_n_distinct'''
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


def engineer_sms(df, day_offset):
    '''
    INPUT: DataFrame with raw SMS log data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
            - participantID
            - date
            - sms_incoming (count)
            - sms_outgoing (count)
            - sms_diff (count equaling (sms_incoming-sms_outgoing))
    '''

    ''' Keeps observations within date range (where a day is defined as 4 AM to 4 AM the next day)'''
    df = _limit_dates(df, day_offset=day_offset)

    ''' Calculates counts of incoming and outgoing texts each day for each participant '''
    df = _calc_incoming_outgoing(df)


    return df


# def engineer_bt_network(df, day_offset):
#     '''
#     INPUT: DataFrame with raw Bluetooth Proximity data
#     OUTPUT: DataFrame--cleaned and engineered. Contains columns:
#             - participantID
#             - date
#             - bt_n
#                 --> Number of devices a participant is within BT proximity of each day
#             - bt_n_distinct
#                 --> Number of distinct devices a participant is within BT proximity of each day
#     '''
#
#     ''' Limits dates to relevant period; removes possibly erroneous nighttime observations'''
#     df = df.rename(columns={'date': 'local_time'})
#     df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
#     ''' Per Friends and Family paper (8.2.1), removes b/n midnight and 7 AM '''
#     df = df[df['local_time'].dt.hour >= 7]
#     df = _limit_dates(df, day_offset=day_offset)
#     df = df[pd.notnull(df['participantID.B'])]
#
#     df_network = df.copy()
#     df_network.loc[:, 'cnt'] = 1
#     df_network_cnts = df_network.groupby(['participantID', 'participantID.B'])['cnt'].count().reset_index()
#     df_network_cnts2 = df_network_cnts.copy()
#     df_network_merged = df_network_cnts.merge(df_network_cnts2, left_on=['participantID', 'participantID.B'],\
#                                         right_on=['participantID.B', 'participantID'])
#     df_network_merged['mean_cnt'] = df_network_merged.mean(axis=1)
#     df_network_merged.rename(columns={'participantID_x': 'participantID.A', 'participantID.B_x': 'participantID.B'}, inplace=True)
#     df_network_merged = df_network_merged[['participantID.A', 'participantID.B', 'mean_cnt']]
#     # ''' DELETE (EDA) '''
#     # sample_users = ['fa10-01-01', 'fa10-01-02', 'fa10-01-03', 'fa10-01-04', 'fa10-01-05', 'fa10-01-06', \
#     #             'fa10-01-07', 'fa10-01-08', 'fa10-01-11', 'fa10-01-12', 'fa10-01-13', 'fa10-01-14', 'fa10-01-15']
#     # for user in sample_users:
#     #     print user, "Mean bt-proximity counts for ", user, "'s closest 5 neighbors:"
#     #     print df_network_merged[df_network_merged['participantID.A'] == user].sort('mean_cnt', ascending=False)['mean_cnt'][0:5]
#     #     print "\n"
#     #
#     # ''' END DELETE '''
#     dnm_sorted = df_network_merged.sort(['participantID.A', 'mean_cnt'], ascending=False)
#
#     ''' Creates bt_innet_top1, bt_innet_2_4, bt_innet_5_10, bt_innet_all'''
#     for user in dnm_sorted['participantID.A'].unique():
#         dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_top1'] = \
#                     sum(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[:1]['mean_cnt'])
#         dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_2_4'] = \
#                     sum(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[1:4]['mean_cnt'])
#         dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_5_10'] = \
#                     sum(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[4:10]['mean_cnt'])
#         dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_all'] = \
#                     sum(dnm_sorted[dnm_sorted['participantID.A'] == user]['mean_cnt'])
#
#     ''' Creates the above but normalized to a per-day basis '''
#     dnm_sorted['n_days_partic'] = dnm_sorted['participantID.A'].map(dict(df.groupby('participantID')['date'].nunique()))
#     dnm_sorted.loc[:, 'bt_innet_top1_perday'] = dnm_sorted['bt_innet_top1'].astype(float) / dnm_sorted['n_days_partic']
#     dnm_sorted.loc[:, 'bt_innet_2_4_perday'] = dnm_sorted['bt_innet_2_4'].astype(float) / dnm_sorted['n_days_partic']
#     dnm_sorted.loc[:, 'bt_innet_5_10_perday'] = dnm_sorted['bt_innet_5_10'].astype(float) / dnm_sorted['n_days_partic']
#     dnm_sorted.loc[:, 'bt_innet_all_perday'] = dnm_sorted['bt_innet_all'].astype(float) / dnm_sorted['n_days_partic']
#     cols_to_drop = ['bt_innet_top1', 'bt_innet_2_4', 'bt_innet_5_10', 'bt_innet_all', 'n_days_partic']
#     dnm_sorted.drop(cols_to_drop, axis=1, inplace=True)
#     print "Per-day columns created."
#
#     #top10 = {}
#     df.loc[:, 'cnt'] = 1
#     for user in dnm_sorted['participantID.A'].unique():
#         #top10[user] = dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[:10]['participantID.B']
#         top10 = list(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[:10]['participantID.B'])
#         top1 = top10[:1]
#         top_2_4 = top10[1:4]
#         top_5_10 = top10[4:10]
#         df_temp = df[df['participantID'] == user]
#         mask1 = df_temp['participantID.B'] == top1[0]
#         mask_2_4 = df_temp['participantID.B'].map(lambda x: top_2_4.count(x) > 0)
#         mask_5_10 = df_temp['participantID.B'].map(lambda x: top_5_10.count(x) > 0)
#
#         mask1_dict = dict(df_temp[mask1].groupby('date')['cnt'].count())
#         mask_2_4dict = dict(df_temp[mask_2_4].groupby('date')['cnt'].count())
#         mask_5_10dict = dict(df_temp[mask_5_10].groupby('date')['cnt'].count())
#         all_dict = dict(df_temp.groupby('date')['cnt'].count())
#
#         df.loc[df['participantID'] == user, 'bt_innet_top1'] = df['date'].map(mask1_dict)
#         df.loc[df['participantID'] == user, 'bt_innet_2_4'] = df['date'].map(mask_2_4dict)
#         df.loc[df['participantID'] == user, 'bt_innet_5_10'] = df['date'].map(mask_5_10dict)
#         df.loc[df['participantID'] == user, 'bt_innet_all'] =  df['date'].map(all_dict)
#
#     print "Daily value columns created."
#     perday_cols = ['bt_innet_top1_perday', 'bt_innet_2_4_perday', 'bt_innet_5_10_perday', 'bt_innet_all_perday']
#     dnm_collapsed = dnm_sorted[perday_cols + ['participantID.A']].drop_duplicates()
#     for col in perday_cols:
#         col_dict = dict(dnm_collapsed[['participantID.A', col]].set_index('participantID.A')[col])
#         df[col] = df['participantID'].map(col_dict)
#     df['bt_innet_top1_pct'] = df['bt_innet_top1'].astype(float) / df['bt_innet_top1_perday']
#     df['bt_innet_2_4_pct'] = df['bt_innet_2_4'].astype(float) / df['bt_innet_2_4_perday']
#     df['bt_innet_5_10_pct'] = df['bt_innet_5_10'].astype(float) / df['bt_innet_5_10_perday']
#     df['bt_innet_all_pct'] = df['bt_innet_all'].astype(float) / df['bt_innet_all_perday']
#
#     df.drop(['participantID.B', 'address', 'cnt'], axis=1, inplace=True)
#     df = df.drop_duplicates().reset_index()
#     df = df[pd.notnull(df['participantID'])]
#     df = df.fillna(0)
#
#     return df

def engineer_call(df, day_offset):
    '''
    INPUT: DataFrame with raw call log data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
            - participantID
            - date
            - call_incoming (count)
            - call_outgoing (count)
            - call_diff (count equaling (call_incoming-call_outgoing))
    '''

    ''' Keeps observations within date range (where a day is defined as 4 AM to 4 AM the next day)'''
    df = _limit_dates(df, day_offset=day_offset)

    ''' Drops missed calls and strips + from outgoing+ and incoming+ '''
    df = df[df['type'] != 'missed']

    df['type'] = df['type'].map(lambda x: str(x).strip('+'))

    ''' Calculates counts of incoming and outgoing texts each day for each participant '''
    df = _calc_incoming_outgoing(df)
    df = df.rename(columns={'sms_diff': 'call_diff', 'sms_incoming': 'call_incoming', \
                            'sms_outgoing': 'call_outgoing', 'sms_total': 'call_total'})
    return df



def engineer_battery(df, day_offset):
    '''
    INPUT: DataFrame with raw battery data
    OUTPUT: DataFrame--cleaned and engineered. Contains columns:
            - participantID
            - date
            - level
            - plugged
            - temperature
            - voltage
        These last 4 are all daily means for each participant.
    '''

    df = df.rename(columns={'date': 'local_time'})  # So can feed into _limit_dates
    # print "df_Battery before limiting dates: df['date'].min() = ", df['date'].min()
    df = _limit_dates(df, day_offset=day_offset)
    # print "df_Battery after limiting dates: df['date'].min() = ", df['date'].min()
    df.loc[df['plugged'] > 1, 'plugged'] = 1
    #df = df.groupby(['participantID', 'date'])[['level', 'plugged', 'temperature', 'voltage']].mean().reset_index()


    ''' Experimenting: gets min, mean, and max of 4 battery feature columns '''
    df_new = df[['participantID', 'date']].drop_duplicates().reset_index().drop('index', axis=1)
    min_mean_max_cols = ['level', 'plugged', 'temperature', 'voltage']
    # new_cols = []
    for col in min_mean_max_cols:
        min_name = col + "_min"
        mean_name = col + "_mean"
        max_name = col + "_max"
        #df[min_name] = df.groupby(['participantID', 'date'])[col].min()
        grouped = df.groupby(['participantID', 'date'])[col]
        df_new[min_name] = grouped.min().reset_index()[col]
        df_new[mean_name] = grouped.mean().reset_index()[col]
        df_new[max_name] = grouped.max().reset_index()[col]
        # new_cols += [min_name, mean_name, max_name]
    df_new.drop(['plugged_min', 'plugged_max'], axis=1, inplace=True)
    # new_cols.remove('plugged_min')
    # new_cols.remove('plugged_max')
    # new_cols += ['date', 'participantID']

    #df_new = df_new[new_cols]
    #df = df.groupby(['participantID', 'date'])[['level', 'plugged', 'temperature', 'voltage']].mean().reset_index()
    ''' end Experimenting '''


    return df_new

def engineer(name, feature_df, day_offset):
    '''
    INPUT: string, DataFrame
    OUTPUT: DataFrame
    Engineers a raw DataFrame and returns it.
    --> name is the name of the raw DataFrame
    'weekend' is a dummy var equal to 1 for Friday, Saturday, and Sunday
    '''
    if name == 'df_SMSLog':
        feature_df = engineer_sms(feature_df, day_offset)
        print "df_SMSLog engineered"

    if name == 'df_CallLog':
        feature_df = engineer_call(feature_df, day_offset)
        print "df_CallLog engineered"

    if name == 'df_Battery':
        feature_df = engineer_battery(feature_df, day_offset)
        print "df_Battery engineered"

    if name == 'df_BluetoothProximity':
        feature_df = engineer_bt(feature_df, day_offset)

        print "df_BluetoothProximity engineered"


    ''' Converts 'date' column to Timestamp if necessary (so merge with df_labels works)'''
    if feature_df['date'][0].__class__.__name__ != 'Timestamp':
        feature_df['date'] = feature_df['date'].map(lambda x: Timestamp(x))


    return feature_df


# mt.feature_label_mat['day_of_week'] = mt.feature_label_mat['date'].map(lambda x: x.dayofweek)
# mt.feature_label_mat.loc[mt.feature_label_mat['day_of_week'] >= 4, 'weekend'] = 1

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

# def engineer_all(df_SMSLog, df_CallLog, df_Battery):
#     '''
#     INPUT: DataFrame, DataFrame, DataFrame
#     OUTPUT: 3 DataFrames, engineered
#     '''
#
#     ''' HERE, WANT TO ITERATE THROUGH LIST OF FEAT_DFS AND CALL engineer(feat_df),
#     which itself will be a function calling the appropriate engineer function
#     '''
#
#     df_SMSLog = engineer_sms(df_SMSLog)
#     print "df_SMSLog engineered"
#     df_CallLog = engineer_call(df_CallLog)
#     print "df_CallLog engineered"
#     df_Battery = engineer_battery(df_Battery)
#     print "df_Battery engineered"
#     return df_SMSLog, df_CallLog, df_Battery

def read_in_as_dfs(text_files):
    '''
    INPUT: List of CSV files to read into pandas
    OUTPUT: None
    Reads in CSV files listed in text_files and stores them in globals()
    '''
    for text_file in text_files:
        input_name = text_file
        df_name = "df_" + text_file.split('.')[0]
        globals()[df_name] = pd.read_csv(text_file)

if __name__ == '__main__':

    all_text_files = ["Accel.csv",
                  "BluetoothProximity.csv",
                  "SMSLog.csv",
                  "AccelAccum.csv",
                  "CallLog.csv",
                  "SurveyBig5.csv",
                  "App.csv",
                  "SurveyCouples.csv",
                  "SurveyWeight.csv",
                  "AppRunning.csv",
                  "Location.csv",
                  "SurveyFriendship.csv",
                  "Battery.csv",
                  "SurveyFromPhone.csv"
                ]

    text_files = [
                  "SMSLog.csv",
                  "CallLog.csv",
                  "Battery.csv"
                  ]

    for text_file in text_files:
        input_name = text_file
        df_name = "df_" + text_file.split('.')[0]
        globals()[df_name] = pd.read_csv(text_file)

    '''Bluetooth data starts 7/10/10 (except a tiny amount in 1/2010, likely an error)'''
    #engineer_bt()



    #read_in_as_dfs(text_files)
    print "finished read_in_as_dfs step"
    df_SMSLog, df_CallLog, df_Battery = engineer_all(df_SMSLog, df_CallLog, df_Battery)

    # df_SMSLog = engineer_sms(df_SMSLog)
    # df_CallLog = engineer_call(df_CallLog)
    # df_Battery = engineer_battery(df_Battery)
