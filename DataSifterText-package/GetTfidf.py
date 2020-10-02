"""vectorize text"""
from autocorrect import Speller
import pandas as pd
import sklearn
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from nltk.corpus import words
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from gensim.summarization.summarizer import summarize

tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()

spell = Speller(lang='en')
stopWords = set(stopwords.words('english'))


def summarizeText(text):
    """summarize text if needed"""
    text_summarized = []
    for txt in text:
        try:
            if (summarize(txt) == ''):
                text_summarized.append(txt)
            else:
                text_summarized.append(summarize(txt))
        except:
            text_summarized.append(txt)

    return text_summarized

def Preprocess(text):
    """preprocess given text"""
    tokens = []
    # tokenize
    for txt in text:
        tokens.append(tokenizer.tokenize(txt))
    # stemmer the text
    stemmed_texts = []
    for token in tokens:
        stemmed_words = []
        for word in token:
            word = re.sub('[^A-z\.]', '', word)
            if word:
                word = stemmed_words.append(stemmer.stem(word))
        stemmed_texts.append(' '.join(stemmed_words))
    return stemmed_texts

def vectorize(first_dataframe, summary):
    """vectorize the given text"""
    clean_texts = first_dataframe
    new_texts = []
    label = clean_texts['label']

    text = clean_texts['text']
    # summarize text if needed
    if summary:
        text = summarizeText(text)

    # get word vector
    corpus = Preprocess(text)
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(corpus)
    df_words = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
    # filter out non-english word
    selected = []
    for i in df_words:
        if i in words.words():
            selected.append(i)
    # get selected word vecs
    df_selected = df_words[selected]
    # get tfidf
    transformer = TfidfTransformer()
    tf = transformer.fit_transform(df_selected.to_numpy())
    df_reduced = pd.DataFrame(tf.toarray(), columns = selected)

    df_reduced['event_orig'] = label
    # get training set for mini-batch
    X_train, X_test, y_train, y_test = train_test_split(df_reduced, label, test_size=0.3, random_state=42, stratify = label)
    X_train = X_train.drop(columns=['event_orig'])

    # clustering
    kmeans = MiniBatchKMeans(n_clusters=5,
                         random_state=42).fit(X_train.to_numpy())

    df_reduced = df_reduced.drop(columns=['event_orig'])

    # get clustered labels
    label_pre = kmeans.predict(df_reduced.to_numpy())

    clean_texts['label_pre'] = label_pre
    clean_texts['label'] = label
    if summary:
        clean_texts['summary'] = text

    data_wrap = pd.DataFrame()
    data_wrap = df_reduced
    data_wrap['label_pre'] = list(clean_texts['label_pre'])
    data_wrap['event_true'] = label
    if summary:
        data_wrap['text_summary'] = text
    data_wrap['text_raw'] = list(clean_texts['text'])
    # return preproccessed dataframe
    return data_wrap