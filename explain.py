import nltk
from nltk.corpus import stopwords
import json
import string
from tqdm import tqdm
from nltk.stem.porter import *
from gensim.test.utils import datapath
import gensim

# nltk.download('stopwords')
stopwords_en = stopwords.words('english')
stemmer = PorterStemmer()

model = gensim.models.fasttext.load_facebook_model(datapath('/Everything/Github/tetql-lake/content/wiki.en.bin'))