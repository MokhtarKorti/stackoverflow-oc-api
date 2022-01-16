import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle


nlp = spacy.load("en_core_web_sm")
pos_list = ["NOUN", "PROPN"]

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stopwords_en = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def pos_cleaner(x, nlp, pos_list):
    doc = nlp(x)
    list_text = []
    for token in doc:
        if(token.pos_ in pos_list):
            list_text.append(token.lemma_)
    joined_text = " ".join(list_text)
    return joined_text

def clean(text):
    corpus = BeautifulSoup(text, "html.parser" ).get_text()
    corpus = re.sub("[^a-zA-Z]", " ", corpus)
    corpus = corpus.lower()
    corpus = pos_cleaner(corpus, nlp, pos_list)
    corpus = word_tokenize(corpus)
    corpus = [w for w in corpus if w not in stopwords_en]
    corpus = [lemmatizer.lemmatize(w) for w in corpus]

    return corpus

def vectorize(text):
    filename_tfidf_vectorizer = './models/tfidf_vectorizer.pkl'
    tfidf_vectorizer = pickle.load(open(filename_tfidf_vectorizer, 'rb'))
    return tfidf_vectorizer.transform(text)

def predict_tags(text):
    filename_rfc_tfidf = './models/rfc_cv_tfidf.pkl'
    rfc_tfidf = pickle.load(open(filename_rfc_tfidf, 'rb'))

    filename_multilabel_bin = './models/multilabel_bin.pkl'
    multilabel_bin = pickle.load(open(filename_multilabel_bin, 'rb'))

    filename_logit_reg_cv_tfidf = './models/logit_reg_cv_tfidf.pkl'
    logistic_reg = pickle.load(open(filename_logit_reg_cv_tfidf, 'rb'))

    text = clean(text)
    text = pd.Series([text])

    vectorized_text = vectorize(text)
    y_pred = rfc_tfidf.predict(vectorized_text)
    res = multilabel_bin.inverse_transform(y_pred)
    res = list({tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
        
    return res