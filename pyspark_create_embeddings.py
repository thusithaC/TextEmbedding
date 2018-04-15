import pickle
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize.api import StringTokenizer
import re
import time

from pyspark.sql import SparkSession
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

from autocorrect.nlp_parser import NLP_COUNTS
from autocorrect.word import Word, common, exact, known, get_case

def spell(word):
    """most likely correction for everything up to a double typo"""
    w = Word(word)
    candidates = (common([word]) or exact([word]) or known([word]) or
                  known(w.typos()) or common(w.double_typos()) or
                  [word])
    correction = max(candidates, key=NLP_COUNTS.get)
    gc.collect()
    return get_case(word, correction)


def tokenize(text_rdd, clean_html=False, tokenizer="twitter", remove_reps=True, spell_correct=True):
    if tokenizer=="stanford":
        tokenizer_obj = StanfordTokenizer()
    elif tokenizer=="twitter":
        tokenizer_obj = TweetTokenizer()
    else:
        tokenizer_obj = StringTokenizer()
    print("Processing {} tokns".format(text_rdd.count()))

    if(remove_reps):
        text_rdd = text_rdd.map(lambda text : re.sub(r'(.)\1{2,}', r'\1\1', text))
    if clean_html:
        text_rdd = text_rdd.map(lambda text:BeautifulSoup(text).get_text())
    tokens_rdd = text_rdd.map(lambda text: TweetTokenizer().tokenize(text))
    if spell_correct:
        tokens_rdd = tokens_rdd.map(lambda tokens: [spell(t) for t in tokens])
        #tokens_rdd = tokens_rdd.map(lambda tokens: [t for t in tokens])

    return tokens_rdd


spark = SparkSession.builder \
             .appName("CreateEmbeddings") \
             .getOrCreate()

start = time.time()
print("hello")
df = spark.read.csv(DATA_FILE, sep=',', escape='"', header=True, inferSchema=True, multiLine=True)
print(df.count())
text_rdd = df.select("comment_text").rdd.map(lambda x: str(x.comment_text))
text_rdd = text_rdd.repartition(16).cache()
#text_rdd.sample(False, 0.01)
tokens_rdd = tokenize(text_rdd)
#text_rdd.saveAsTextFile("/home/thusitha/work/PycharmProjects/TextEmbedding/sample_txt_5_percent.txt")
tokens_list = tokens_rdd.collect()
pickle.dump(tokens_list, open("tokens_uncorrected.pickle", 'wb'))
end = time.time()
print(end - start)