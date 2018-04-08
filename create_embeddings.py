import pickle

EMBEDDING_FILE = '/media/SharedData/work/data/pretrained/crawl-300d-2M_trunc.vec.pickle'

if EMBEDDING_FILE.split(".")[-1] =="pickle":
    embeddings = pickle.load(open(EMBEDDING_FILE, "rb"))
else:
    embeddings = open(EMBEDDING_FILE, 'r')


