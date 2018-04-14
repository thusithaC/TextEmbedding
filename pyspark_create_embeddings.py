import pickle
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import StanfordTokenizer
from nltk.tokenize.api import StringTokenizer
import re
from autocorrect import spell
import time

from pyspark.sql import SparkSession

EMBEDDING_FILE = '/media/SharedData/work/data/pretrained/crawl-300d-2M_trunc.vec.pickle'
DATA_FILE = '/media/SharedData/work/data/toxic/train.csv'

def tokenize(text_rdd, clean_html=False, tokenizer="twitter", remove_reps=True, spell_correct=False):
    if tokenizer=="stanford":
        tokenizer_obj = StanfordTokenizer()
    elif tokenizer=="twitter":
        tokenizer_obj = TweetTokenizer()
    else:
        tokenizer_obj = StringTokenizer()

    if(remove_reps):
        text_rdd = text_rdd.map(lambda text : re.sub(r'(.)\1{2,}', r'\1\1', text))
    if clean_html:
        text_rdd = text_rdd.map(lambda text:BeautifulSoup(text).get_text())
    tokens_rdd = text_rdd.map(lambda text: tokenizer_obj.tokenize(text))
    if spell_correct:
        tokens_rdd = tokens_rdd.map(lambda tokens: [spell(t) for t in tokens])

    return tokens_rdd


spark = SparkSession.builder \
             .master("local[*]") \
             .appName("Word Count") \
             .config("spark.some.config.option", "some-value") \
             .getOrCreate()

start = time.time()
print("hello")
df = spark.read.csv(DATA_FILE, header=True)
print(df.count())
text_rdd = df.select("comment_text").rdd.map(lambda x: str(x.comment_text))
text_rdd.sample(False, 0.01)
tokens_rdd = tokenize(text_rdd)
#text_rdd.saveAsTextFile("/home/thusitha/work/PycharmProjects/TextEmbedding/sample_txt_5_percent.txt")
tokens_list = tokens_rdd.collect()
pickle.dump(tokens_list, open("tokens.pickle", 'wb'))
end = time.time()
print(end - start)