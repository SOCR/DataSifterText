import torch
import pandas as pd
import csv

from transformers import AutoTokenizer, AutoModelForMaskedLM

def impute(text_input, label_input, model_name, mask_id, mask_token, cls_token, sep_token):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForMaskedLM.from_pretrained(model_name)

	labels = label_input
	texts = text_input
	predict_texts = []

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
			text, texts[i] = tmp_str, tmp_str

			inputs = tokenizer(text, return_tensors="pt")
			predictions = model(**inputs)

			input_ids = (inputs['input_ids'].tolist())[0]
			if mask_id not in input_ids:
				indices = []
				prev_sent_indices = []
			else:
				indices = [p for p, x in enumerate(input_ids) if x == mask_id]
				prev_sent_indices = [q for q, x in enumerate(text.split()) if x == mask_token]

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
					sort_result = torch.sort(predictions.logits[0,each_index])[1]
					final_result = [] # final_result stores the top 20 possible imputed choices.
					for j in range(20):
						curr_item = tokenizer.convert_ids_to_tokens([sort_result[-j-1].item()])
						if curr_item[0] not in ['.', ',', '-', ';', '?', '!', '|']:
							final_result += [curr_item]
					# We can choose the top 1 or sample from these 20 choices. Here we just choose the top one.
					predict_result += [final_result[0]]
				else:
					# Do not impute this [MASK] since it is a neighbor of previous imputed [MASK]
					repeat_flag = True
					predict_result += [[mask_token]]

			
			words = text.split()
			result = ""
			if not repeat_flag:
				# No [MASK] left, ready to output result
				for k in range(len(words)):
					if words[k] == cls_token or words[k] == sep_token:
						continue
					elif k not in prev_sent_indices:
						result += words[k]
					else:
						result += predict_result[prev_sent_indices.index(k)][0].upper()
					result += ' '
				predict_texts.append(result)
			else:
				# There is still [MASK] left, prepare for next iteration
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
	        'label': labels,
	        'alpha': ['a']*len(final_text_arr),
	        'text': final_text_arr
	    }) 	

	return df_bert
