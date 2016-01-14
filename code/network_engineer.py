import pandas as pd
import networkx as nx
from feature_engineer import _limit_dates

def _graph_centrality_measures(df):
    df = df[df['participantID.A'] > df['participantID.B']]
    G = nx.from_pandas_dataframe(df, 'participantID.A', 'participantID.B', 'mean_cnt')

    degree_centrality = nx.degree_centrality(G)
    eigen_centrality = nx.eigenvector_centrality(G)
    eigen_centrality_weighted = nx.eigenvector_centrality(G, weight='mean_cnt')

    return degree_centrality, eigen_centrality, eigen_centrality_weighted

def engineer_bt_network(df, day_offset):
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
    df = df[pd.notnull(df['participantID.B'])]

    df_network = df.copy()
    df_network.loc[:, 'cnt'] = 1
    df_network_cnts = df_network.groupby(['participantID', 'participantID.B'])['cnt'].count().reset_index()
    df_network_cnts2 = df_network_cnts.copy()
    df_network_merged = df_network_cnts.merge(df_network_cnts2, left_on=['participantID', 'participantID.B'],\
                                        right_on=['participantID.B', 'participantID'])
    df_network_merged['mean_cnt'] = df_network_merged.mean(axis=1)
    df_network_merged.rename(columns={'participantID_x': 'participantID.A', 'participantID.B_x': 'participantID.B'}, inplace=True)
    df_network_merged = df_network_merged[['participantID.A', 'participantID.B', 'mean_cnt']]
    # ''' DELETE (EDA) '''
    # sample_users = ['fa10-01-01', 'fa10-01-02', 'fa10-01-03', 'fa10-01-04', 'fa10-01-05', 'fa10-01-06', \
    #             'fa10-01-07', 'fa10-01-08', 'fa10-01-11', 'fa10-01-12', 'fa10-01-13', 'fa10-01-14', 'fa10-01-15']
    # for user in sample_users:
    #     print user, "Mean bt-proximity counts for ", user, "'s closest 5 neighbors:"
    #     print df_network_merged[df_network_merged['participantID.A'] == user].sort('mean_cnt', ascending=False)['mean_cnt'][0:5]
    #     print "\n"
    #
    # ''' END DELETE '''
    dnm_sorted = df_network_merged.sort(['participantID.A', 'mean_cnt'], ascending=False)




    ''' Graph centrality measures for each participant'''
    degree_centrality, eigen_centrality, eigen_centrality_weighted = _graph_centrality_measures(dnm_sorted)





    ''' Creates bt_innet_top1, bt_innet_2_4, bt_innet_5_10, bt_innet_all'''
    for user in dnm_sorted['participantID.A'].unique():
        dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_top1'] = \
                    sum(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[:1]['mean_cnt'])
        dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_2_4'] = \
                    sum(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[1:4]['mean_cnt'])
        dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_5_10'] = \
                    sum(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[4:10]['mean_cnt'])
        dnm_sorted.loc[dnm_sorted['participantID.A'] == user, 'bt_innet_all'] = \
                    sum(dnm_sorted[dnm_sorted['participantID.A'] == user]['mean_cnt'])

    ''' Creates the above but normalized to a per-day basis '''
    dnm_sorted['n_days_partic'] = dnm_sorted['participantID.A'].map(dict(df.groupby('participantID')['date'].nunique()))
    dnm_sorted.loc[:, 'bt_innet_top1_perday'] = dnm_sorted['bt_innet_top1'].astype(float) / dnm_sorted['n_days_partic']
    dnm_sorted.loc[:, 'bt_innet_2_4_perday'] = dnm_sorted['bt_innet_2_4'].astype(float) / dnm_sorted['n_days_partic']
    dnm_sorted.loc[:, 'bt_innet_5_10_perday'] = dnm_sorted['bt_innet_5_10'].astype(float) / dnm_sorted['n_days_partic']
    dnm_sorted.loc[:, 'bt_innet_all_perday'] = dnm_sorted['bt_innet_all'].astype(float) / dnm_sorted['n_days_partic']
    cols_to_drop = ['bt_innet_top1', 'bt_innet_2_4', 'bt_innet_5_10', 'bt_innet_all', 'n_days_partic']
    dnm_sorted.drop(cols_to_drop, axis=1, inplace=True)
    print "Network's per-day columns created. Creating daily value columns..."

    #top10 = {}
    df.loc[:, 'cnt'] = 1
    for user in dnm_sorted['participantID.A'].unique():
        #top10[user] = dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[:10]['participantID.B']
        top10 = list(dnm_sorted[dnm_sorted['participantID.A'] == user].iloc[:10]['participantID.B'])
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

        df.loc[df['participantID'] == user, 'bt_innet_top1'] = df['date'].map(mask1_dict)
        df.loc[df['participantID'] == user, 'bt_innet_2_4'] = df['date'].map(mask_2_4dict)
        df.loc[df['participantID'] == user, 'bt_innet_5_10'] = df['date'].map(mask_5_10dict)
        df.loc[df['participantID'] == user, 'bt_innet_all'] =  df['date'].map(all_dict)

    print "Daily value columns created."
    perday_cols = ['bt_innet_top1_perday', 'bt_innet_2_4_perday', 'bt_innet_5_10_perday', 'bt_innet_all_perday']
    dnm_collapsed = dnm_sorted[perday_cols + ['participantID.A']].drop_duplicates()
    for col in perday_cols:
        col_dict = dict(dnm_collapsed[['participantID.A', col]].set_index('participantID.A')[col])
        df[col] = df['participantID'].map(col_dict)
    df['bt_innet_top1_pct'] = df['bt_innet_top1'].astype(float) / df['bt_innet_top1_perday']
    df['bt_innet_2_4_pct'] = df['bt_innet_2_4'].astype(float) / df['bt_innet_2_4_perday']
    df['bt_innet_5_10_pct'] = df['bt_innet_5_10'].astype(float) / df['bt_innet_5_10_perday']
    df['bt_innet_all_pct'] = df['bt_innet_all'].astype(float) / df['bt_innet_all_perday']

    df.drop(['participantID.B', 'address', 'cnt'], axis=1, inplace=True)
    df = df.drop_duplicates().reset_index()
    df = df[pd.notnull(df['participantID'])]

    ''' Graph centrality characteristics '''
    df.loc[:, 'degree_centrality'] = df['participantID'].map(degree_centrality)
    df.loc[:, 'eigen_centrality'] = df['participantID'].map(eigen_centrality)
    df.loc[:, 'eigen_centrality_weighted'] = df['participantID'].map(eigen_centrality_weighted)

    df.fillna(0, inplace=True)

    print "Network df engineered"
    return df
