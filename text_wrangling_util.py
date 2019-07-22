"""
Methods for processing the textual input data
"""
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

stop_words = []
with open('nltk_english_stopwords.txt', 'r') as f:
    for line in f:
        stop_words.append(line.split()[0])

def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]"," ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def stemming(text):
    stemmed_text = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemmed_text += stem
        stemmed_text += " "
    stemmed_text = stemmed_text.strip()
    return stemmed_text

def prepare_input_text(title, description):
    if type(title) == str:
        title = [title]
    if type(description) == str:
        description = [description]

    input_text = [i+" "+v for (i,v) in zip(title,description)]
    input_text = [clean_text(i) for i in input_text]
    input_text = [remove_stopwords(i) for i in input_text]
    input_text = [stemming(i) for i in input_text]

    return input_text
