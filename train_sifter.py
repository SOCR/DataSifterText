import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

X = []
y = []
with open('data/trainData.csv', 'r') as f:
	reader = csv.reader(f)
	i = 0
	for row in reader:
		i += 1
		if i == 1:
			continue
		if row[3] in ['62', '42', '55', '63', '71']:
			X += [row[0:3]]
			y += row[3:]
		# if i == 500:
		# 	break

print(len(X))
print(len(y))

df_bert_total = pd.DataFrame({
	'id': range(len(X)),
	'text': [i[0] for i in X]
	})
df_bert_total.to_csv('data/test-total.tsv', sep='\t', index=False, header=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
print(len(X_test))
print(len(y_test))


df_bert = pd.DataFrame({
	'id':range(len(X_train)),
	'label':y_train,
	'alpha':['a']*len(X_train),
	'text':[i[0] for i in X_train]
	})

print(df_bert.head())

df_bert_train, df_bert_dev = train_test_split(df_bert, test_size = 0.11111)

print(len(df_bert_dev))
print(len(df_bert_train))

df_bert_test = pd.DataFrame({
	'id':range(len(X_test)),
	'text':[i[0] for i in X_test]
	})

print(df_bert_test.head())

df_bert_test_result = pd.DataFrame({
	'label': y_test
	})

df_bert_train.to_csv('data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=True)
df_bert_test_result.to_csv('data/test_result.tsv', sep='\t', index=False, header=True)



