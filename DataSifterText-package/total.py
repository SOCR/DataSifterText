from preprocess_0821 import process
from mask import mask
from predict_tsv_1_seq import impute
from refine_label import refine_label
from csv_to_tsv import to_tsv
from GetTfidf import vectorize
from swap import Obfuscate
import pandas as pd
import nltk
import time
import argparse

# starting time
start = time.time()

nltk.download('words')

# receive parameteres from users
def main(args):
	if args.model_mode == 'bert':
		MODEL_NAME = 'bert-base-uncased'
	elif args.model_mode == 'bert_large':
		MODEL_NAME = 'bert-large-uncased'
	elif args.model_mode == 'electra':
		MODEL_NAME = 'google/electra-base-generator'
	elif args.model_mode == 'pubmedbert':
		MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
		MASK_ID = 4
	
	if args.model_mode in ['bert', 'bert_large', 'electra']:
		MASK_ID = 103
	if args.model_mode in ['bert', 'bert_large',  'electra', 'pubmedbert']:
		MASK_TOKEN = '[MASK]'
		CLS_TOKEN = '[CLS]'
		SEP_TOKEN = '[SEP]'

	SUMMARIZE = args.summarize  # 0: no summarize, 1: summarize
	KEYWORDS_POSITION = args.keywords_position # 0: keywords, 1: position
	FILE_NAME = args.file_name

	dataframe = pd.read_csv(FILE_NAME)
	rows = refine_label(dataframe)
	rows = process(rows)
	text, label = mask(rows, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN)
	first_dataframe = impute(text, label, MODEL_NAME, MASK_ID, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN)

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

	# end time
	end = time.time()

	# total time taken
	# print(f"Runtime of the program is {end - start}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--model_mode", default='pubmedbert', help="name of model type to use", required=True, choices=['bert', 'bert_large', 'electra', 'pubmedbert'])
	parser.add_argument("--summarize", type=int, default=0, help="0: no summarize, 1: summarize", required=True, choices=[0, 1])
	parser.add_argument("--keywords_position", type=int, default=0, help="0: keywords, 1: position", required=True, choices=[0, 1])
	parser.add_argument("--file_name", default='', help="file name", required=True)
	args = parser.parse_args()
	main(args)