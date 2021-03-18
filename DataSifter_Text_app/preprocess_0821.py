import csv
import re
from autocorrect import Speller
import sys

def process(rows_input):
    spell = Speller(lang='en')
    rows = []

    for case in rows_input:
        text, label = case

        # Remove characters that are not in alphabets, numerics or space
        text = text.replace('\n', " ")
        tmp = list([val for val in text if val.isalpha() or val==' ' or val == ',' or val == '.'])
        text = "".join(tmp)

        if text == " ":
            continue

        # All lower case
        text = text.lower()

        # Autocorrect for each word
        words = text.split()
        for k in range(len(words)):
            words[k] = spell(words[k])
        tmp = ""
        if len(words) == 0:
            continue
        for k in range(len(words)-1):
            tmp += words[k]
            tmp += " "
        tmp += words[-1]
        text = tmp
        rows.append([text, label])

    return rows
