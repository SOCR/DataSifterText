import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import random
import nltk
from nltk.corpus import stopwords
import pandas as pd
from string import ascii_lowercase


random.seed(42)
nltk.download('stopwords')

blacklist = set(stopwords.words('english'))
blacklist.add('dx')

for c in ascii_lowercase[1:]:
    blacklist.add(c)

whitelist = set(
["pain", "neck", "strain", "work", "lift", "head", "back", "cervic", "fell", "foot", "knee", "ankl",
"hi", "hit", "eye", "hand", "handyom", "cut", "metal", "lacer", "lac", "burn", "low", "lower", "finger",
"stuck", "shoulder"])

def mask(rows):
    X = []
    y = []
    idx = []
    masked_X = []

    for case in rows:
        text, label = case
        X.append(text)
        y.append(label)

    for i in range(len(X)):
        curr = X[i]
        curr = curr.split()
        cnt = 0
        coeffi = 1.2
        for j in range(len(curr)):
            random_num = random.uniform(0,1)
            if random_num > 0.75 * coeffi:
                if curr[j] not in blacklist:
                    curr[j] = '[MASK]'
                    cnt += 1
                    coeffi = 1.2
            elif curr[j] in whitelist and random_num > 0.5 * coeffi:
                curr[j] = '[MASK]'
                cnt += 1
                coeffi = 1.2
            else:
                coeffi -= 0.05

            if j > 512:
                break
            if cnt >= min(len(curr), 512) * 0.5:
                break

        change_sentence = ""
        for j in range(len(curr)):
            change_sentence += curr[j]
            if j != (len(curr) - 1):
                change_sentence += " "
            if j > 420:
                break
        X[i] = change_sentence
        X[i] = '[CLS] ' + X[i] + " [SEP]"

    return X, y