import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
import csv

def impute():
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	id_list = []
	labels = []
	texts = []
	predict_texts = []
	with open('masked_f1.tsv','r') as f:
		read_tsv = csv.reader(f, delimiter="\t")
		for row in read_tsv:
			id_list.append(row[0])
			labels.append(row[1])
			texts.append(row[3])

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


			last_index = -2
			predict_result = []
			for each_index in indices:
				if last_index + 1 != each_index:
					sort_result = torch.sort(predictions[0,each_index])[1]
					final_result = []
					for j in range(20):
						curr_item = tokenizer.convert_ids_to_tokens([sort_result[-j-1].item()])
						if curr_item[0] in ['.', ',', '-', ';', '?', '!', '|']:
							pass
						else:
							final_result += [curr_item]
					predict_result += [final_result[0]]
				else:
					repeat_flag = True
					predict_result += [['[MASK]']]
				last_index = each_index

			if not repeat_flag:
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
	        'id':id_list,
	        'label':labels,
	        'alpha':['a']*len(final_text_arr),
	        'text': final_text_arr
	    })

	df_bert.to_csv('bert_test_f1_seq.tsv', sep='\t', index = False, header = True)


if __name__ == "__main__":
	impute()
