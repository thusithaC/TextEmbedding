import pickle
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize.api import StringTokenizer
import re
from autocorrect import spell
import platform
import gc

DESKTOP_PC = 'thusitha-MS-7A34'

if platform.node() == DESKTOP_PC:
    print("Desktop PC, 8 processors")
    NUM_PROC = 5
    EMBEDDING_FILE = '/home/thusitha/work/bigdata/datasets/pretrained/crawl-300d-2M_trunc.vec.pickle'
    DATA_FILE = '/home/thusitha/work/bigdata/datasets/toxic/train.csv'
else:
    NUM_PROC = 4
    EMBEDDING_FILE = '/media/SharedData/work/data/pretrained/crawl-300d-2M_trunc.vec.pickle'
    DATA_FILE = '/media/SharedData/work/data/toxic/train.csv'



def getdata_from_csv(csvpath):
    return pd.read_csv(csvpath)

def get_embeddings(embedding_file:str):
    if embedding_file.split(".")[-1] =="pickle":
        embeddings = pickle.load(open(embedding_file, "rb"))
    else:
        embeddings = open(embedding_file, 'r')
    return embeddings

def tokenize(text_list, clean_html=False, tokenizer="twitter", remove_reps=True, spell_correct=True):
    if tokenizer=="stanford":
        tolkenizer_obj = StanfordTokenizer()
    elif tokenizer=="twitter":
        tolkenizer_obj = TweetTokenizer()
    else:
        tolkenizer_obj = StringTokenizer()

    token_list = []
    for text in text_list:
        if clean_html:
            text = BeautifulSoup(text).get_text()
        if remove_reps:
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        tokens = tolkenizer_obj.tokenize(text)
        if spell_correct:
            tokens = [spell(t) for t in tokens]
        token_list.append(tokens)
    return token_list

def __tolkenize_text_blob(text, clean_html, remove_reps, spell_correct, tolkenizer_obj):
    if clean_html:
        text = BeautifulSoup(text).get_text()
    if remove_reps:
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    tokens = tolkenizer_obj.tokenize(text)
    if spell_correct:
        tokens = [spell(t) for t in tokens]
    gc.collect()
    return tokens

def par_tokenize(text_list, clean_html=False, tokenizer="twitter", remove_reps=True, spell_correct=True):
    if tokenizer=="stanford":
        tolkenizer_obj = StanfordTokenizer()
    elif tokenizer=="twitter":
        tolkenizer_obj = TweetTokenizer()
    else:
        tolkenizer_obj = StringTokenizer()

    import multiprocessing as mp
    from functools import partial
    pool = mp.Pool(NUM_PROC)
    tolkenize_func = partial(__tolkenize_text_blob, clean_html=clean_html, remove_reps=remove_reps, spell_correct=spell_correct, tolkenizer_obj=tolkenizer_obj)
    token_list = pool.map(tolkenize_func, text_list)
    return token_list


df = getdata_from_csv(DATA_FILE)
tokens_list  = tokenize(df.comment_text, clean_html=False)
#tokens_list  = par_tokenize(df.comment_text, clean_html=False)
pickle.dump(tokens_list, open( "tokens_list.pickle", "wb" ) )


