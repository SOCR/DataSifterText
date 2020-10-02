from preprocess_0821 import process
from mask import mask
from predict_tsv_1_seq import impute
from refine_label import refine_label
from csv_to_tsv import to_tsv
import sys

# receive parameteres from users
SUMMARIZE = 0  # 0: no summarize, 1: summarize
KEYWORDS_POSITION = 0 # 0: keywords, 1: position

SUMMARIZE = sys.argv[1]
KEYWORDS_POSITION = sys.argv[2]

refine_label()
process("0")
mask()
impute()
to_tsv()

if SUMMARIZE:
	# SUMMARIZE FUNC
	print("Summarized text and obfuscate")
else:
	# NO SUMMARIZE FUNC
	if KEYWORDS_POSITION:
		# Position FUNC
		print("no summarization, doing pos-swap")
	else:
		# Keywords Func
		print("no summarization, doing keywords-swap")

