import math
import os
import warnings
import time
import datetime
import json
import bz2
import pandas as pd
import numpy as np


def keywords_search(keywords=None, df=None, save_mode=False, save_filename=None):
    '''
    Perform a keywords search on the DataFrame df and return the processed DataFrame.
    If df=None: load all the data from the quotes.json.bz2 files (folder ../Datasets)
    If df!=None: perform the keywords search on the provided DataFrame
    Note: if df=None, quotes-YYYY.json.bz2 (from quotebank data) must be in the folder ../Datasets/ to use keywords_search function with df=None.
    :param keywords: [list of string] keywords search in the quotes
    :param df: [optional, DataFrame] DataFrame containing the quotes
    :param save_mode: [bool, optional] save the new DataFrame in save_filename
    :param save_filename: [string, optional] filename for the save of the DataFrame
    :return: [DataFrame] processed DataFrame with the quotes containing at least one keyword | None returned if problem occurs
    '''
    if keywords is None:
        keywords = {'climate change'}
    if save_filename is None:
        save_filename = "quotes_keyworded.pkl"
    time_begin = time.time()
    n_quotes = 0
    if df is None:
        if os.path.isdir("../Datasets") == False:
            warnings.warn("The quotes can't be loaded ! The folder ../Datasets does not exist. None returned by keywords_search function.")
            return None
        for year in range(2008, 2021):
            if os.path.isfile(f"../Datasets/quotes-{year}.json.bz2") == False:
                warnings.warn(f"The quotes from {year} can't be loaded. The file ../Datasets/quotes-{year}.json.bz2 does not exist.")
            else:
                with bz2.open(f"../Datasets/quotes-{year}.json.bz2", 'rb') as f:
                    quotes_kept = {}
                    for line in f:
                        if n_quotes % 100000 == 0: print(f"{n_quotes} quotes processed (year {year}) | ({round(time.time() - time_begin, 2)} sec)")
                        n_quotes += 1
                        quote = json.loads(line)
                        quote_useful = False
                        for keyword in keywords:
                            if quote['quotation'].lower().find(keyword) != -1: quote_useful = True
                        if quote_useful:
                            quotes_kept.update({len(quotes_kept): quote})
                df_chunk = pd.DataFrame.from_dict(quotes_kept, orient='index')
                if df is None:
                    df = df_chunk
                else:
                    df = df.append(df_chunk)
    else:
        masks = None
        for keyword in keywords:
            if masks is None:
                masks = (df['quotation'].str.lower().str.find(keyword) != -1).to_frame(name=keyword)
            else:
                masks[keyword] = (df['quotation'].str.lower().str.find(keyword) != -1)
        global_mask = masks.any(axis=1)
        df = df[global_mask]
    if save_mode:
        df.to_pickle(save_filename)
    print(f"Total time for keywords_search function: {round(time.time() - time_begin, 2)} sec")
    return df


def add_speakers_attributes(df, save_mode=False, save_filename=None):
    '''
    Add the speakers attributes to the DataFrame df provided and return the result.
    Note: speaker_attributes.parquet (from the provided Drive from ADA course) must be in the folder ../Datasets/ to use add_speakers_attributes function.
    :param df: [DataFrame] DataFrame containing the quotes
    :param save_mode: [bool, optional] save the new DataFrame in save_filename
    :param save_filename: [string, optional] filename for the save of the DataFrame
    :return: [DataFrame] processed DataFrame with the quotes containing the speakers attributes
    '''
    if save_filename is None:
        save_filename = "quotes_with_speakers.pkl"
    time_begin = time.time()
    if os.path.isdir("../Datasets"):
        if os.path.isdir("../Datasets/speaker_attributes.parquet"):
            pass
        else:
            warnings.warn("speaker_attributes.parquet (from the provided Drive from ADA course) must be in the folder ../Datasets/ to use add_speakers_attributes function. None returned by add_speakers_attributes function.")
            return None
        pass
    else:
        warnings.warn("speaker_attributes.parquet (from the provided Drive from ADA course) must be in the folder ../Datasets/ to use add_speakers_attributes function. None returned by add_speakers_attributes function.")
        return None
    speakers = pd.read_parquet("../Datasets/speaker_attributes.parquet", engine='pyarrow')
    name_mapper = {}
    for old_name in list(speakers.columns):
        new_name = "speaker_" + old_name
        name_mapper.update({old_name:new_name})
    name_mapper.update({'id':'qid'})
    speakers.rename(columns=name_mapper, inplace=True)
    speaker_columns = list(speakers.columns)
    speaker_columns.remove('qid')
    speaker_columns.append('speaker_age')
    for speaker_col in speaker_columns:
        df[speaker_col] = [[] for _ in range(df.shape[0])]

    chunksize = 1000
    for chunk in range(math.ceil(df.shape[0]/chunksize)):
        min_bound = chunk*chunksize
        max_bound = (chunk+1)*chunksize
        if max_bound > df.shape[0]: max_bound = df.shape[0]
        qids_used = []
        for qids in list(df.iloc[min_bound:max_bound]['qids']):
            for qid in qids:
                qids_used.append(qid)
        speakers_mask = speakers['qid'].isin(qids_used)
        speakers_chunk = speakers[speakers_mask]

        for i in range(min_bound, max_bound):
            if i % 1000 == 0: print(f"{i}/{df.shape[0]} quotes with speaker added | ({round(time.time() - time_begin, 2)} sec)")
            for qid in df.iloc[i]['qids']:
                speaker = speakers_chunk.loc[(speakers_chunk['qid'] == qid)]
                for speaker_col in speaker_columns:
                    if speaker.shape[0] == 1:
                        if speaker_col == 'speaker_date_of_birth':
                            date_of_birth = speaker['speaker_date_of_birth'].item()
                            if date_of_birth is not None:
                                for j in range(len(date_of_birth)):
                                    date_of_birth[j] = date_of_birth[j].replace('Z','').replace('T',' ').replace('+','')
                                    if len(date_of_birth[j]) != 19:
                                        warnings.warn(f"This error should not arise, check the code [invalid date of birth: {date_of_birth[j]}]")
                            df.iloc[i][speaker_col].append(date_of_birth)
                        elif speaker_col == 'speaker_age':
                            date_of_birth = speaker['speaker_date_of_birth'].item()
                            if date_of_birth is None:
                                age = None
                            elif len(date_of_birth) > 0:
                                birth_datetime = []
                                for j in range(len(date_of_birth)):
                                    try:
                                        birth_datetime.append(datetime.datetime.strptime(date_of_birth[j], '%Y-%m-%d %H:%M:%S'))
                                    except:
                                        pass
                                if len(birth_datetime) > 0:
                                    invalid_birth = False
                                    for b in birth_datetime:
                                        if abs((birth_datetime[0]-b).total_seconds()) > 320000000: invalid_birth = True
                                    if invalid_birth == False:
                                        try:
                                            quote_datetime = datetime.datetime.strptime(df.iloc[i]['date'], '%Y-%m-%d %H:%M:%S')
                                            age_seconds = 0.0
                                            for k in range(len(birth_datetime)):
                                                age_seconds += (quote_datetime - birth_datetime[k]).total_seconds()/len(birth_datetime)
                                            age = round(age_seconds / (86400 * 365), 2)
                                            if age > 125: age = "NOT_KNOWN"
                                        except:
                                            age = "NOT_KNOWN"
                                    else:
                                        age = "NOT_KNOWN"
                                else:
                                    age = "NOT_KNOWN"
                            else:
                                warnings.warn(f"This error should not arise, check the code [invalid date_of_birth for one speaker {speaker['qid'].item()}: {date_of_birth}]")
                                age = "NOT_KNOWN"
                            df.iloc[i][speaker_col].append(age)
                        else:
                            df.iloc[i][speaker_col].append(speaker[speaker_col].item())
                    elif speaker.shape[0] == 0:
                        df.iloc[i][speaker_col].append("NOT_FOUND")
                    else:
                        warnings.warn(f"This error should not arise, check the code [multiple speakers with same qid: {speaker['qid']}]")
    if save_mode:
        df.to_pickle(save_filename)
    print(f"Total time for add_speakers_attributes function: {round(time.time() - time_begin, 2)} sec")
    return df


def get_label_from_QID(QID):
    '''
    Get the label (string) associated with a QID
    :param QID: [string] QID
    :return: [string] <label of QID> or <QID> if label not found
    '''
    qids = pd.read_pickle("../Datasets/qids.pkl")
    label = qids[qids.QID==QID]
    if label.shape[0] == 0:
        return QID
    else:
        return label.iloc[0]['Label']



