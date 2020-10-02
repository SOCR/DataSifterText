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
nltk.download('words')

# receive parameteres from users
SUMMARIZE = 0  # 0: no summarize, 1: summarize
KEYWORDS_POSITION = 0 # 0: keywords, 1: position

SUMMARIZE = int(sys.argv[1])
KEYWORDS_POSITION = int(sys.argv[2])
FILE_NAME = sys.argv[3]

dataframe = pd.read_csv(FILE_NAME)
rows = refine_label(dataframe)
rows = process(rows)
text, label = mask(rows)
first_dataframe = impute(text, label)


if SUMMARIZE:
	# SUMMARIZE FUNC
	tfidf_result = vectorize(first_dataframe, True)
	Obfuscate(tfidf_result, "summary")
else:
	# NO SUMMARIZE FUNC
	if KEYWORDS_POSITION:
		# Position FUNC
		tfidf_result = vectorize(first_dataframe, False)
		Obfuscate(tfidf_result, "pos")
	else:
		# Keywords Func
		tfidf_result = vectorize(first_dataframe, False)
		Obfuscate(tfidf_result, "keyword")

