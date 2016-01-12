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

    df = df.rename(columns={'date': 'local_time'})

    df['local_time'] = pd.DatetimeIndex(pd.to_datetime(df['local_time']))
    ''' Per Friends and Family paper (8.2.1), removes b/n midnight and 7 AM '''
    df = df[df['local_time'].dt.hour >= 7]

    df = _limit_dates(df, day_offset=day_offset)



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


    # df = df.rename(columns={'address': 'bt_n'})
    # df = df.groupby(['participantID', 'date'])['address'].nunique().reset_index()
    # df = df.rename(columns={'address': 'bt_n_distinct'})

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
