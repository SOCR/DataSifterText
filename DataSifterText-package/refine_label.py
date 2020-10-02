import csv
import pandas as pd

def refine_label(dataframe):
	rows = []

	for i in range(len(dataframe)):
		text = dataframe['text'][i]
		label = dataframe['label'][i]
		if label == '62':
			label = 0
		elif label == '42':
			label = 1
		elif label == '55':
			label = 2
		elif label == '63':
			label = 3
		elif label == '71':
			label = 4
		rows.append([text, label])

	return rows
