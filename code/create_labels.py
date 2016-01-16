import pandas as pd
import string

def _read_clean(fname):
    '''
    INPUT: string
    OUTPUT: DataFrame

    Reads in CSV file and returns a cleaned DataFrame.
    '''
    input_name = '../data/' + fname
    df_phone_survey = pd.read_csv(input_name)
    df_mood_qs = df_phone_survey[df_phone_survey['questions_raw'].map(lambda x: string.find(x, 'happy') >= 0)]
    df_mood_qs.loc[:, 'answer_date'] = df_mood_qs['date'].map(lambda x: pd.to_datetime(x.split()[0]))
    df_mood_qs.loc[:, 'date'] = df_mood_qs['name'].map(lambda x: pd.to_datetime(x.split()[1]))
    ''' Drops rows where not all answers are filled in'''
    df_mood_qs.loc[:, 'answers_len'] = df_mood_qs['answers_raw'].map(lambda x: len(x))
    df_mood_qs = df_mood_qs[((df_mood_qs['answers_len'] >= 42) & (df_mood_qs['answers_len'] <= 44))] # sic
    return df_mood_qs

def _extract_mood_responses(df_mood_qs):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame

    Extracts participants' numerical rankings of their daily moods.
    '''
    df_mood_qs['happy'] = df_mood_qs['answers_raw'].map(lambda x: int(x[17]))
    df_mood_qs['stressed'] = df_mood_qs['answers_raw'].map(lambda x: int(x[21]))
    df_mood_qs['productive'] = df_mood_qs['answers_raw'].map(lambda x: x[25])
    df_mood_qs = df_mood_qs[df_mood_qs['productive'] != '>']    # Drops very few
    df_mood_qs.loc[:, 'productive'] = df_mood_qs['productive'].map(lambda x: int(x))
    df_mood_qs.drop(['name', 'answers_len', 'questions_raw', 'answers_raw'], axis=1, inplace=True)
    return df_mood_qs

def _create_dummies(df_mood_qs, to_dummyize, very_cutoff_inclusive, very_un_cutoff_inclusive):
    '''
    INPUT: DataFrame, list, int, int
    OUTPUT: DataFrame

    Creates 3 dummies for any/all moods in to_dummyize:
        - [mood]_dummy, 1 if mood rated as >= 5
        - very_[mood], 1 if mood rated as >= very_cutoff_inclusive
        - very_un[mood], 1 if mood rated as <= very_un_cutoff_inclusive
    '''
    for lab in to_dummyize:
        dummy_name = lab + '_dummy'
        very_name = 'very_' + lab
        very_un_name = 'very_un' + lab
        df_mood_qs[dummy_name] = 0 + (df_mood_qs[lab] >= 5)
        df_mood_qs[very_name] = 0 + (df_mood_qs[lab] >= 6)
        df_mood_qs[very_un_name] = 0 + (df_mood_qs[lab] <= 2)
    return df_mood_qs

def create_poss_labels(fname, to_dummyize, very_cutoff_inclusive=6, very_un_cutoff_inclusive=2, answer_offset_cutoff=-1):
    '''
    INPUT: string, int
    OUTPUT: DataFrame

    Returns a DataFrame whose columns correspond to labels to be predicted.

    --> answer_offset_cutoff: if != -1, answers submitted *at least* answer_offset_cutoff
        days after the date in question will be omitted

    Columns as follows (indented are optional dummies specified by to_dummyize parameter):
        - participantID
        - date
        - happy (1-7, 7 is happiest)
            - happy_dummy, 1 if happy >= 4
            - very_happy, 1 if happy >= very_cutoff_inclusive
            - very_unhappy, 1 if happy <= very_un_cutoff_inclusive
        - stressed (1-7, 7 is most stressed)
            - stressed_dummy, 1 if stressed >= 4
            - very_stressed, 1 if stressed >= very_cutoff_inclusive
            - very_unstressed, 1 if stressed <= very_un_cutoff_inclusive
        - productive (1-7, 7 is most productive)
            - productive_dummy, 1 if productive >= 4
            - very_productive, 1 if productive >= very_cutoff_inclusive
            - very_unproductive, 1 if productive <= very_un_cutoff_inclusive
    '''
    df_mood_qs = _read_clean(fname)
    df_mood_qs = _extract_mood_responses(df_mood_qs)
    df_mood_qs = _create_dummies(df_mood_qs, to_dummyize, very_cutoff_inclusive, very_un_cutoff_inclusive)

    ''' Drops where the survey is answered before the corresponding date has passed '''
    df_mood_qs['answer_offset_days'] = df_mood_qs['answer_date'] - df_mood_qs['date']
    df_mood_qs = df_mood_qs[df_mood_qs['answer_offset_days'] >= pd.to_timedelta('0 days')]
    if answer_offset_cutoff != -1:
        df_mood_qs = df_mood_qs[df_mood_qs['answer_offset_days'] < answer_offset_cutoff]
    df_mood_qs.drop(['answer_offset_days', 'answer_date'], axis=1, inplace=True)

    return df_mood_qs
