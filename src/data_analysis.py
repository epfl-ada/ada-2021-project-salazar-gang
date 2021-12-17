import math
import os
import warnings
import time
import datetime
import json
import bz2
import pandas as pd
import numpy as np


def find_keywords_in_label(df, label="speaker_occupation", keywords=['researcher']):
    '''
    Return a reduced version of "df" where "df[label]" contains (at least) one keyword from "keywords"
    :param df: [DataFrame] Original DataFrame
    :param label: [str] Label of the (DataFrame) column where the keywords are searched
    :param keywords: [list] Keywords searched in df[label]
    :return: DataFrame where df[label] contains (at least) one keyword
    '''
    if type(keywords) is not list: keywords = [keywords]
    if os.path.isfile("../Datasets/qids.pkl"):
        qids = pd.read_pickle("../Datasets/qids.pkl")
    else:
        warnings.warn("../Datasets/qids.pkl does not exist !")
        return None
    qids_keyworded = qids[qids['Label'].astype(str).str.lower().str.contains('|'.join(keywords),case=False)]
    qids_keyworded = qids_keyworded['QID'].tolist()
    mask = df[label].astype(str).str.contains('|'.join(qids_keyworded),case=False)
    df_keyworded = df[mask]
    return df_keyworded


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


def keep_only_first_speaker(df):
    df = df[df['speaker_type'].str.len()==1]
    return df





