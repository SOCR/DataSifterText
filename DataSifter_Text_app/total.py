from preprocess_0821 import process
from mask import mask
from predict_tsv_1_seq import impute
from refine_label import refine_label
from csv_to_tsv import to_tsv
from GetTfidf import vectorize
from swap import Obfuscate
import pandas as pd
import sys
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('words')

def compute(dataframe):

    #nltk.download('words')

    # receive parameteres from users
    SUMMARIZE = 0  # 0: no summarize, 1: summarize
    KEYWORDS_POSITION = 0 # 0: keywords, 1: position

    #SUMMARIZE = sys.argv[1]
    #KEYWORDS_POSITION = sys.argv[2]
    rows = refine_label(dataframe)
    rows = process(rows)
    text, label = mask(rows)
    first_dataframe = impute(text, label)
    print("ready to swap!")

    if SUMMARIZE:
        # SUMMARIZE FUNC
        tfidf_result = vectorize(dataframe, True)
        result = Obfuscate(tfidf_result, "summary")
    else:
        # NO SUMMARIZE FUNC
        if KEYWORDS_POSITION:
            # Position FUNC
            tfidf_result = vectorize(dataframe, False)
            result = Obfuscate(tfidf_result, "pos")
        else:
            # Keywords Func
            tfidf_result = vectorize(dataframe, False)
            result = Obfuscate(tfidf_result, "keyword")
    return result

