#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: text_processing.py
Date: November 2023
"""

import os
import re
import nltk
import spacy
import pandas as pd
from textblob import TextBlob
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

PATH = '/Users/xzmarquez/Documents/GitHub/Personal/chicago-mayoral-election-2023/final-project/data'
nlp = spacy.load("en_core_web_sm")


# Text Processing cocde focusing on cleaning up the text, making it lowercase,
# removing stopwords, applying sentiment analysis, and subjectivity score
def data_file_path(filename):
    return os.path.join(PATH, filename)


def remove_punctuation(raw_text):
    """
    Removes punctuation from the text.
    """
    pattern = re.compile(r'[^\w\s]')
    clean_text = re.sub(pattern, '', raw_text).strip()
    return clean_text


def process_text(raw_text):
    """
    Process raw text by changing the text to lowercase,
    stripping leading and whitespace, tokenizing and removing stop words.

    Parameters
    ----------
        raw_text : the raw text (string), either an open end or transcript

    Returns
    -------
        clean_text : the cleaned text of an open end
    """
    stop_words = set(stopwords.words('english'))
    clean_text = raw_text.lower().strip()
    clean_text = remove_punctuation(clean_text)
    clean_text = word_tokenize(clean_text)  # tokenize
    clean_text = [w for w in clean_text if w not in stop_words]
    return clean_text


def get_wordnet_pos(treebank_tag):
    """
    Returns the WordNet part of speech for each Tree Bank tag.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return


def calculate_subjectivity(text):
    """
    Using TextBlob to determine the media source's subjectivity on candidate
    """
    blob = TextBlob(text)
    return blob.sentiment.subjectivity


def get_polarity_score(text):
    """
    Using TextBlob to determine polarity score
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


clean_news_df = pd.read_csv(data_file_path('clean_news_sources.csv'))

clean_news_df['clean_title'] = clean_news_df['Title'].apply(remove_punctuation)
clean_news_df['clean_title'] = clean_news_df['clean_title'].apply(process_text)
clean_news_df['wordnet_pos'] = clean_news_df['clean_title'].apply(
    lambda x: [get_wordnet_pos(tag) for word, tag in nltk.pos_tag(x)])

# Join the processed words in the 'clean_title' column back into a string
# so we can apply the sentiment analysis score and subjectivity score
clean_news_df['clean_title'] = clean_news_df['clean_title'].apply(
    lambda x: ' '.join(x) if isinstance(x, list) else x)

clean_news_df['sentiment_score'] = clean_news_df['clean_title'].apply(
    lambda x: SentimentIntensityAnalyzer().polarity_scores(str(x))['compound'])

clean_news_df['subjectivity_score'] = clean_news_df['clean_title'].apply(
    calculate_subjectivity)
clean_news_df['polarity_score'] = clean_news_df['clean_title'].apply(
    get_polarity_score)

##############################################################################
# In this part of the code, we are going to be looking at endorsements the
# candidates recieved and looking at certain words associated with these
# endorsements using spaCy


def contains_endorsement(title):
    """
    Checks if the given title contains any endorsement-related keywords.
    """
    endorsement_keywords = ['endorses', 'endorsed', 'endorsement']
    return any(keyword in title.lower() for keyword in endorsement_keywords)


def tokenize_and_remove_stopwords_spacy(text):
    """
    Tokenizes the input text using spaCy, converting each token to
    lowercase and removing stop words.

    Parameters:
    text (str): The text to tokenize.

    Returns:
    list: A list of lowercase tokens without stopwords.
    """
    doc = nlp(text)
    filtered_tokens = [
        token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return filtered_tokens


def extract_linguistic_features(text):
    """
    Extracts linguistic features (lowercase token and part-of-speech tag)
    from the input text using spaCy,
    considering only certain parts of speech (ADJ, NOUN, VERB).

    Returns:
    list: A list of tuples containing the lowercase token and part-of-speech tag.
    """
    doc = nlp(text)
    features = [(
        token.text.lower(), token.pos_) for token in doc if token.pos_ in ('ADJ', 'NOUN', 'VERB')]
    return features


def mentions_brandon_johnson(title):
    """
    Checks if the lowercase version of the title mentions "brandon johnson."

    """
    return 'brandon johnson' in title.lower()


def mentions_paul_vallas(title):
    """
    Checks if the lowercase version of the title mentions "paul vallas."

    """
    return 'paul vallas' in title.lower()


endorsement_rows = clean_news_df[clean_news_df['Title'].apply(contains_endorsement)]

selected_columns = ['Title', 'Source', 'clean_title', 'subjectivity_score', 'polarity_score']
selected_df = clean_news_df[selected_columns].copy()

brandon_johnson_rows = selected_df[selected_df['Title'].apply(
    mentions_brandon_johnson)].reset_index(drop=True)
paul_vallas_rows = selected_df[selected_df['Title'].apply(
    mentions_paul_vallas)].reset_index(drop=True)

# Apply the extraction function to each title mentioning Johnson and Vallas
brandon_johnson_rows['linguistic_features'] = brandon_johnson_rows[
    'clean_title'].apply(extract_linguistic_features)

paul_vallas_rows['linguistic_features'] = paul_vallas_rows[
    'clean_title'].apply(extract_linguistic_features)

########### Finding the average subjectivity and polarity score #############
mean_subjectivity_johnson = brandon_johnson_rows['subjectivity_score'].mean()
mean_polarity_johnson = brandon_johnson_rows['polarity_score'].mean()

mean_subjectivity_vallas = paul_vallas_rows['subjectivity_score'].mean()
mean_polarity_vallas = paul_vallas_rows['polarity_score'].mean()

print("Mean Subjectivity for Johnson:", mean_subjectivity_johnson)
print("Mean Polarity for Johnson:", mean_polarity_johnson)

print("Mean Subjectivity for Vallas:", mean_subjectivity_vallas)
print("Mean Polarity for Vallas:", mean_polarity_vallas)
