import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
import csv

def impute(text_input, label_input):

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	labels = label_input
	texts = text_input
	predict_texts = []

	model = BertForMaskedLM.from_pretrained('bert-base-uncased')
	model.eval()

	for i in range(len(texts)):
		repeat_flag = True
		next_predict_text = texts[i]
		while repeat_flag:
			repeat_flag = False
			text = next_predict_text
			words = text.split()[:290]
			tmp_str = ""
			for word_idx in range(len(words)):
				tmp_str += words[word_idx]
				tmp_str += " "
			texts[i] = tmp_str
			text = texts[i]
			tokenized_text = tokenizer.tokenize(text)
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

			# Create the segments tensors.
			segments_ids = [0] * len(tokenized_text)

			# Convert inputs to PyTorch tensors
			tokens_tensor = torch.tensor([indexed_tokens])
			segments_tensors = torch.tensor([segments_ids])

			# Predict all tokens
			with torch.no_grad():
			    predictions = model(tokens_tensor, segments_tensors)
			if '[MASK]' not in tokenized_text:
				indices = []
				prev_sent_indices = []
			else:
				indices = [p for p, x in enumerate(tokenized_text) if x == '[MASK]']
				prev_sent_indices = [q for q, x in enumerate(text.split()) if x == '[MASK]']

			# indices and prev_sent_indicies store the positions of [MASK]
			# indices: store the tokenized [MASK] position (for imputation)
			# prev_sent_indicies: store the original [MASK] position (for final output)

			# Impute for missingness ([MASK]), do not impute neighborhood [MASK]
			# last_index: last imputed index. If last_index + 1 == each_index (current index), it means
			# 		we are imputing a neighborhood [MASK]
			# last_index = None
			predict_result = []
			for each_index in indices:
				if each_index + 1 not in indices:    				
				# if (last_index is None) or (last_index + 1 != each_index):
					# impute this [MASK]
					sort_result = torch.sort(predictions[0,each_index])[1]
					final_result = [] # final_result stores the top 20 possible imputed choices.
					for j in range(20):
						curr_item = tokenizer.convert_ids_to_tokens([sort_result[-j-1].item()])
						if curr_item[0] not in ['.', ',', '-', ';', '?', '!', '|']:
							# pass
							final_result += [curr_item]
						# else:
							# final_result += [curr_item]
					# We can choose the top 1 or sample from these 20 choices. Here we just choose the top one.
					predict_result += [final_result[0]]
				else:
					# Do not impute this [MASK] since it is a neighbor of previous imputed [MASK]
					repeat_flag = True
					predict_result += [['[MASK]']]
				# last_index = each_index

			if not repeat_flag:
								

				# No [MASK] left, ready to output result
				words = text.split()
				result = ""

				for k in range(len(words)):
					if words[k] == '[CLS]' or words[k] == '[SEP]':
						continue
					elif k not in prev_sent_indices:
						result += words[k]
					else:
						result += predict_result[prev_sent_indices.index(k)][0].upper()
					result += ' '
				predict_texts.append(result)
			else:

				# There is still [MASK] left, prepare for next iteration
				words = text.split()
				result = ""
				for k in range(len(words)):
					if k not in prev_sent_indices:
						result += words[k]
					else:
						result += predict_result[prev_sent_indices.index(k)][0].upper()
					result += ' '
				next_predict_text = result

	final_text_arr = []
	for each in predict_texts:
		final_text_arr.append(each.lower())

	df_bert = pd.DataFrame({
	        'id': range(len(labels)),
	        'label':labels,
	        'alpha':['a']*len(final_text_arr),
	        'text': final_text_arr
	    }) 	

	return df_bert





