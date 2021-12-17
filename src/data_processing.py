import math
import os
import warnings
import time
import datetime
import json
import bz2
import pandas as pd
import numpy as np
from transformers import pipeline


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
        df.to_pickle("../Datasets/"++save_filename)
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
                                    if len(date_of_birth[j]) != 19 and len(date_of_birth[j]) != 20:
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
                                            if 'date' in df.columns:
                                                quote_datetime = datetime.datetime.strptime(df.iloc[i]['date'], '%Y-%m-%d %H:%M:%S')
                                            else:
                                                quote_datetime = datetime.datetime.strptime('2015-04-01 00:00:00', '%Y-%m-%d %H:%M:%S') # ~ median date of the QuoteBank dataset
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
        df.to_pickle("../Datasets/"+save_filename)
    print(f"Total time for add_speakers_attributes function: {round(time.time() - time_begin, 2)} sec")
    return df


def compute_sentiment_analysis(quotes, model_name="distilbert-base-uncased-finetuned-sst-2-english", classifier_parameters=[]):
    '''
    Computes sentiment analysis on quotes
    :param quotes: [list] quotes
    :param model_name: [str] model used (from huggingface)
    :return: predictions from the model
    '''
    pd.options.mode.chained_assignment = None  # default='warn'
    if model_name == "distilbert-base-uncased-finetuned-sst-2-english":
        classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
    elif model_name == "bhadresh-savani/distilbert-base-uncased-emotion":
        classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
    elif model_name == "facebook/bart-large-mnli":
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        if len(classifier_parameters) == 0: classifier_parameters = [['optimism', 'pessimism']]
    else:
        warnings.warn("Sentiment analysis model unknown !")
        classifier = pipeline("text-classification", model=model_name, return_all_scores=True)

    pred = classifier(["labels wanted !"], *classifier_parameters)
    if type(pred[0]) is dict:
        if pred[0].get('labels') != None:
            pred = [[{'label': pred[0]['labels'][i], 'score': pred[0]['scores'][i]} for i in range(len(pred[0]['labels']))]]
    labels = []
    for l in range(len(pred[0])):
        label = pred[0][l]['label']
        labels.append(label)

    prediction = []
    for i in range(len(quotes)):
        quote = quotes[i]
        try:
            pred = classifier([quote], *classifier_parameters)
            if type(pred[0]) is dict:
                if pred[0].get('labels') != None:
                    pred = [[{'label': pred[0]['labels'][i], 'score': pred[0]['scores'][i]} for i in range(len(pred[0]['labels']))]]
        except:
            pred = [[]]
            for label in labels:
                pred[0].append({'label':label, 'score':np.nan})
        prediction.append(pred[0])
    return prediction


def add_sentiment_analysis(df, model_name="distilbert-base-uncased-finetuned-sst-2-english", classifier_parameters=[], save_mode=False, save_filename=None):
    '''

    :param df:
    :param model_name:
    :param classifier_parameters:
    :param save_mode:
    :param save_filename:
    :return:
    '''
    pd.options.mode.chained_assignment = None  # default='warn'
    if os.path.isdir("sa_checkpoints") is False: os.mkdir("sa_checkpoints")

    prediction = compute_sentiment_analysis(["labels wanted !"], model_name=model_name, classifier_parameters=classifier_parameters)
    for l in range(len(prediction[0])):
        label = "sa_" + prediction[0][l]['label'].lower()
        df[label] = np.nan

    t0 = time.time()
    t_delta = t0
    chunksize = 10000
    for chunk in range(math.ceil(df.shape[0]/chunksize)):
        min_bound = chunk*chunksize
        max_bound = (chunk+1)*chunksize
        if max_bound > df.shape[0]: max_bound = df.shape[0]
        quotes = df.iloc[min_bound:max_bound]['quotation'].values
        quotes = quotes.tolist()
        prediction = compute_sentiment_analysis(quotes, model_name=model_name, classifier_parameters=classifier_parameters)
        for q in range(len(prediction)):
            for l in range(len(prediction[q])):
                label = "sa_"+prediction[q][l]['label'].lower()
                proba = prediction[q][l]['score']
                df[label].iat[min_bound+q] = proba
        print(f"Number of quotes processed with sentiment analysis: {max_bound}/{df.shape[0]} ({round(time.time()-t0,2)} sec)")
        if save_mode and time.time()-t_delta>3600:
            t_delta = time.time()
            timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', 'h')[:16]
            df.to_pickle(f"../Datasets/sa_checkpoints/quotes_sa_checkpoint_{timestamp}.pkl")
    if save_mode:
        df.to_pickle("../Datasets/"+save_filename)
    return df


def get_quotes_per_speaker(keywords=None, debug=False, save_mode=False, save_filename=None):
    pd.options.mode.chained_assignment = None  # default='warn'
    if keywords is None:
        keywords = {'climate change'}
    # keywords = pd.DataFrame([list(keywords)])
    keywords = list(keywords)
    print(keywords)
    if os.path.isdir("../Datasets") == False:
        warnings.warn("The quotes can't be loaded ! The folder ../Datasets does not exist. None returned by keywords_search function.")
        return None
    speakers = pd.DataFrame(columns=['name', 'qids', 'n_all', 'n_climate'])
    t0 = time.time()
    for year in range(2008, 2021):
        if os.path.isfile(f"../Datasets/quotes-{year}.json.bz2") == False:
            warnings.warn(f"The quotes from {year} can't be loaded. The file ../Datasets/quotes-{year}.json.bz2 does not exist.")
        else:
            chunksize = 100000
            df_reader = pd.read_json(f"../Datasets/quotes-{year}.json.bz2", compression='bz2', lines=True, chunksize=chunksize)
            for df_chunk in df_reader:
                df_chunk.loc[:,'name'] = df_chunk['probas'].map(lambda x: x[0][0])
                speakers_chunk = df_chunk[['name','qids']]
                speakers_chunk.loc[:,'name'] = speakers_chunk.loc[:,'name'].str.lower() # lower case speaker name for compatibility
                climate_related = df_chunk.quotation.str.contains('|'.join(keywords),case=False).astype(int) # boolean (0,1) indicating if "climate change relating quote"
                speakers_chunk.loc[:,'n_all'] = 1 - climate_related.values
                speakers_chunk.loc[:,'n_climate'] = climate_related.values
                speakers_chunk.loc[:, 'qids_str'] = speakers_chunk.loc[:, 'qids'].str.join(',') # qids as string for the groupby method

                speakers = pd.concat([speakers,speakers_chunk])
                speakers.loc[:,'n_all'] = speakers.groupby(['name'])['n_all'].transform('sum')
                speakers.loc[:,'n_climate'] = speakers.groupby(['name'])['n_climate'].transform('sum')
                speakers = speakers.drop_duplicates(subset=['name'])

                if debug: print(f"Processed quotes: {speakers['n_all'].sum()+speakers['n_climate'].sum()} ({round(time.time()-t0,2)} sec)")

                # print(speakers.shape,speakers['n_all'].sum(),speakers['n_climate'].sum())  # debugging line
                # print(speakers[speakers.duplicated(subset=['name'])])  # debugging line
                # print(speakers.iloc[0:20])  # debugging line
    if save_mode:
        speakers.to_pickle("../Datasets/"+save_filename)


def find_qids(df):
    if os.path.isfile("../Datasets/wikidata_labels_descriptions.csv.bz2") == False:
        warnings.warn("../Datasets/wikidata_labels_descriptions.csv.bz2 does not exist !")
        return None
    t0 = time.time()
    print("Loading of wikidata_labels_descriptions...")
    QIDS = pd.read_csv("../Datasets/wikidata_labels_descriptions.csv.bz2", compression='bz2', on_bad_lines='warn')
    print(f"Loading done ! ({round(time.time()-t0,2)} sec)")

    # FIND USED QIDS
    chunk = 20
    QIDS_used = set()
    t0 = time.time()
    for i in range(0, df.shape[0], chunk):
        df_str = df.iloc[i:i + chunk].to_string()
        while df_str.find('Q') != -1:
            qid = 'Q'
            df_str = df_str[df_str.find('Q') + 1:]
            begin = True
            while df_str[0] in "0123456789":
                if df_str[0]=='0' and begin:
                    df_str = df_str[1:]
                    continue
                else:
                    begin = False
                    qid += df_str[0]
                    df_str = df_str[1:]
            if len(qid) > 1: QIDS_used.add(qid)
        if (i + chunk) % 1000 == 0 or i + chunk > df.shape[0]: print(f"QIDS searching | Chunk {min(i + chunk, df.shape[0])}/{df.shape[0]}: {round(time.time() - t0, 2)} sec")
        # if (i + chunk) % 10000 == 0: json.dump(list(QIDS_used), open("qids_used.txt", mode='w'))
    # json.dump(list(QIDS_used), open("qids_used.txt", mode='w'))
    print(f"Number of different QIDS used in df: {len(QIDS_used)}")

    QIDS_useful = QIDS[QIDS['QID'].isin(list(QIDS_used))]
    QIDS_useful = QIDS_useful.drop_duplicates(subset=['QID'], keep='first')
    QIDS_useful.to_pickle("../Datasets/qids.pkl")


def create_climate_quotes():
    pd.options.mode.chained_assignment = None  # default='warn'
    keywords = {"climate change", "global warming", "greenhouse effect", "greenhouse gas", "climate crisis", "climate emergency", "climate breakdown"}
    try:
        keywords_search(keywords=keywords, save_mode=True, save_filename="quotes1_keywords.pkl")
        print("keywords_search: GOOD")
    except:
        print("keywords_search: ERROR")
    try:
        df = pd.read_pickle("../Datasets/quotes1_keywords.pkl")
        add_speakers_attributes(df, save_mode=True, save_filename="quotes2_speakers.pkl")
        print("add_speakers_attributes: GOOD")
    except:
        print("add_speakers_attributes: ERROR")
    df = pd.read_pickle("../Datasets/quotes2_speakers.pkl")
    find_qids(df)
    try:
        df = pd.read_pickle("../Datasets/quotes2_speakers.pkl")
        add_sentiment_analysis(df, model_name="distilbert-base-uncased-finetuned-sst-2-english", save_mode=True, save_filename="quotes3_sa1.pkl")
        print("add_sentiment_analysis (1): GOOD")
    except:
        print("add_sentiment_analysis (1): ERROR")
    df = pd.read_pickle("../Datasets/quotes3_sa1.pkl")
    df['sa_score'] = df[['sa_negative', 'sa_positive']].idxmax(axis=1).str[3:]
    df.to_pickle("../Datasets/quotes3_sa1.pkl")
    try:
        df = pd.read_pickle("../Datasets/quotes3_sa1.pkl")
        add_sentiment_analysis(df, model_name="bhadresh-savani/distilbert-base-uncased-emotion", save_mode=True, save_filename="quotes4_sa.pkl")
        print("add_sentiment_analysis (2): GOOD")
    except:
        print("add_sentiment_analysis (2): ERROR")
    df = pd.read_pickle("../Datasets/quotes4_sa.pkl")
    df['sa_emotion'] = df[['sa_sadness', 'sa_joy', 'sa_love', 'sa_anger', 'sa_fear', 'sa_surprise']].idxmax(axis=1).str[3:]
    df.to_pickle("../Datasets/quotes-climate_v2.pkl")

    try:
        get_quotes_per_speaker(keywords, debug=True, save_mode=True, save_filename="quotes_densities.pkl")
        print("get_quotes_per_speaker: GOOD")
    except:
        print("get_quotes_per_speaker: ERROR")
    try:
        df = pd.read_pickle("../Datasets/quotes_densities.pkl")
        add_speakers_attributes(df, save_mode=True, save_filename="quotes_densities_speakers.pkl")
        print("add_speakers_attributes densities: GOOD")
    except:
        print("add_speakers_attributes densities: ERROR")


if __name__ =='__main__':
    create_climate_quotes()








