"""Swapper"""

import pandas as pd
import scipy
import numpy as np
import math
from rake_nltk import Rake
import concurrent.futures


group0 = pd.DataFrame()
group1 = pd.DataFrame()
group2 = pd.DataFrame()
group3 = pd.DataFrame()
group4 = pd.DataFrame()
METHOD = "" 

def GetKeywords(text):
    r = Rake()
    keywords_rank = []
    for sentence in text :
        r.extract_keywords_from_text(sentence)
        keywords_rank.append(r.get_ranked_phrases_with_scores())
        
    top_words = []
    for key_ls in keywords_rank:
        if not key_ls:
            top_words.append(' ')
        else:
            top_words.append(key_ls[0][1])
    return top_words

def Swap(index, tfidf):
    cos_si = []
    ran_sam = tfidf.iloc[index]
    label_find = int(ran_sam['label_pre'])
    if label_find == 0:
            compare_pool = group0.sample(n = 1000)
            compute_pool = compare_pool.to_numpy()
    if label_find == 1:
            compare_pool = group1.sample(n = 1000)
            compute_pool = compare_pool.to_numpy()
    
    if label_find == 2:
            compare_pool = group2.sample(n = 1000)
            compute_pool = compare_pool.to_numpy()
            
    if label_find == 3:
            compare_pool = group3.sample(n = 1000)
            compute_pool = compare_pool.to_numpy()
            
    if label_find == 4:
            compare_pool = group4.sample(n = 1000)
            compute_pool = compare_pool.to_numpy()
    
    for row in compute_pool:
        if METHOD == "summary":
            compute_sam = ran_sam.drop(columns = ['event_true','text_raw','text_summary'])
        else:
            compute_sam = ran_sam.drop(columns = ['event_true','text_raw','rake_keywords'])
        length = len(tfidf.columns) - 4
        compute_row = row[0:length]
        n1 = np.linalg.norm(compute_sam.to_numpy()[0:length])
        n2 = np.linalg.norm(compute_row)
        cos_si.append(np.dot(compute_row,compute_sam.to_numpy()[0:length]) / (n1*n2))
        
    compare_pool['cos_distance'] = cos_si
    compare_pool = compare_pool.sort_values(by=['cos_distance'], ascending=False)
    r = math.ceil(0.1*len(compare_pool))
    if r > 1:
        if METHOD == "summary":
            data_select = compare_pool.iloc[1:r]
            swap_text = data_select.sample(n = 1)
        
            text_be_swapped = ran_sam['text_raw']
            
            #find original summary
            summary_orig = ran_sam['text_summary'].split('.')
            summary_dest = ''
            for txt in summary_orig:
                if txt != '':
                    summary_dest = txt
                    break

            text_swap = list(swap_text['text_raw'])[0]
            summary_swap = list(swap_text['text_summary'])[0].split('.')
            summary_source = ''
            for txt in summary_swap:
                if txt != '':
                    summary_source = txt
                    break
            
            new_text = text_be_swapped.replace(summary_dest, summary_source)

        elif METHOD == "pos":
            data_select = compare_pool.iloc[1:r]
            swap_text = data_select.sample(n = 1)
        
            text_be_swapped = ran_sam['text_raw']
            text_swap = list(swap_text['text_raw'])[0]
            index_start = text_swap.find(list(swap_text['rake_keywords'])[0])
            
            index_start0 = text_be_swapped.find(ran_sam['rake_keywords'])
            
            new_text = text_be_swapped.replace(text_be_swapped[index_start0:], list(swap_text['text_raw'])[0][index_start:])

        else:
            data_select = compare_pool.iloc[1:r]
            swap_text = data_select.sample(n = 1)
        
            text_be_swapped = ran_sam['text_raw']
        
            new_text = text_be_swapped.replace(ran_sam['rake_keywords'], list(swap_text['rake_keywords'])[0])

        return new_text
    else:
        return ran_sam['text_raw']

def Obfuscate(tfidf, method):
    """Swap Driver"""
    METHOD = method
    tfidf = tfidf.drop(columns= [tfidf.keys()[0]])
    text = tfidf['text_raw']
    if method == "summary":
        summary = list(tfidf['text_summary'])
        new_summary = []
        for sry in summary:
            new_summary.append(sry.replace('\n', ''))
        tfidf['text_summary'] = new_summary

    else: 
        tfidf['rake_keywords'] = GetKeywords(text)

    copy_of_sample = tfidf

    group0 = copy_of_sample[copy_of_sample['label_pre'] == 0]
    group1 = copy_of_sample[copy_of_sample['label_pre'] == 1]
    group2 = copy_of_sample[copy_of_sample['label_pre'] == 2]
    group3 = copy_of_sample[copy_of_sample['label_pre'] == 3]
    group4 = copy_of_sample[copy_of_sample['label_pre'] == 4]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        index = list(range(0, len(tfidf)))
        texts = executor.map(Swap, index, tfidf)

    obfuscated_text = pd.DataFrame()
    obfuscated_text['raw_text'] = tfidf['text_raw']
    obfuscated_text['swapped_text'] = texts
    obfuscated_text['label'] = tfidf['event_true']
    output = 'Obfuscated_?.csv', (METHOD,)
    obfuscated_text.to_csv(output)

