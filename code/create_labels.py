import pandas as pd
import string


def create_poss_labels(fname, answer_offset_cutoff=-1):
    '''
    INPUT: string, int
    OUTPUT: DataFrame
    Returns a DataFrame with possible labels.
    --> answer_offset_cutoff: if != -1, answers submitted *at least* answer_offset_cutoff
        days after the date in question will be omitted
    Columns as follows:
        - participantID
        - date
        - answer_date (the date the participant answered the survey on his/her phone)
        - happy (1-7, 7 is happiest)
            - happy_dummy, 1 if happy >= 4
            - very_happy, 1 if happy >= 6
            - very_unhappy, 1 if happy <= 2
        - stressed (1-7, 7 is most stressed)
            - stressed_dummy, 1 if stressed >= 4
            - very_stressed, 1 if stressed >= 6
            - very_unstressed, 1 if stressed <= 2
        - productive (1-7, 7 is most productive)
            - productive_dummy, 1 if productive >= 4
            - very_productive, 1 if productive >= 6
            - very_unproductive, 1 if productive <= 2
        - answer_offset_days (equal to (answer_date - date))
    '''
    input_name = '../data/' + fname
    df_phone_survey = pd.read_csv(input_name)
    df_mood_qs = df_phone_survey[df_phone_survey['questions_raw'].map(lambda x: string.find(x, 'happy') >= 0)]
    df_mood_qs.loc[:, 'answer_date'] = df_mood_qs['date'].map(lambda x: pd.to_datetime(x.split()[0]))
    #df_mood_qs.drop('date', axis=1, inplace=True)

    df_mood_qs.loc[:, 'date'] = df_mood_qs['name'].map(lambda x: pd.to_datetime(x.split()[1]))
    #df_mood_qs.drop('name', axis=1, inplace=True)

    ''' Drops rows where not all answers are filled in'''
    df_mood_qs.loc[:, 'answers_len'] = df_mood_qs['answers_raw'].map(lambda x: len(x))
    df_mood_qs = df_mood_qs[((df_mood_qs['answers_len'] >= 42) & (df_mood_qs['answers_len'] <= 44))] # sic
    #df_mood_qs.drop('answers_len', axis=1, inplace=True)

    ''' '''
    df_mood_qs['happy'] = df_mood_qs['answers_raw'].map(lambda x: int(x[17]))
    df_mood_qs['stressed'] = df_mood_qs['answers_raw'].map(lambda x: int(x[21]))
    df_mood_qs['productive'] = df_mood_qs['answers_raw'].map(lambda x: x[25])
    df_mood_qs = df_mood_qs[df_mood_qs['productive'] != '>']    # Drops very few
    df_mood_qs['productive'] = df_mood_qs['productive'].map(lambda x: int(x))
    df_mood_qs.drop(['name', 'answers_len', 'questions_raw', 'answers_raw'], axis=1, inplace=True)

    ''' "Label engineers": creates dummies, e.g., happy_dummy (>=4), very_happy (>=6), very_unhappy (<=2)'''
    for lab in ['happy']:#, 'stressed', 'productive']:
        dummy_name = lab + '_dummy'
        very_name = 'very_' + lab
        very_un_name = 'very_un' + lab
        df_mood_qs[dummy_name] = 0 + (df_mood_qs[lab] >= 4)
        df_mood_qs[very_name] = 0 + (df_mood_qs[lab] >= 6)
        df_mood_qs[very_un_name] = 0 + (df_mood_qs[lab] <= 2)

    ''' Drops where the survey is answered before the corresponding date has passed '''
    df_mood_qs['answer_offset_days'] = df_mood_qs['answer_date'] - df_mood_qs['date']
    df_mood_qs = df_mood_qs[df_mood_qs['answer_offset_days'] >= pd.to_timedelta('0 days')]
    if answer_offset_cutoff != -1:
        df_mood_qs = df_mood_qs[df_mood_qs['answer_offset_cutoff'] < answer_offset_cutoff]
    df_mood_qs.drop(['answer_offset_days', 'answer_date'], axis=1, inplace=True)

    return df_mood_qs


if __name__ == '__main__':
    df = create_poss_labels('SurveyFromPhone.csv')
