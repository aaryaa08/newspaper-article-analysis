#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install newspaper3k

import nltk
import newspaper
nltk.download('punkt')
from newspaper import Article
article_name = Article("https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/", language="en")
article_name.download()
article_name.parse()
article_name.nlp()
article_name.text
article_name.title


print(article_name.title)
 

print(article_name.text)
article=article_name.text


# In[3]:


print(article_name.summary)


# In[5]:


print(article_name.keywords)


# In[7]:


import pandas as pd
import numpy as np
data = pd.read_csv('./StopWords.txt', error_bad_lines=False)
data


# In[8]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words=data
text_tokens = word_tokenize(article)
tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
final_article=" ".join(tokens_without_sw)
print(final_article)


# In[9]:


positive_words = pd.read_csv('./positive-words.txt', error_bad_lines=False)
negative_words = pd.read_csv('./negative-words.txt', error_bad_lines=False)


# In[10]:


import spacy
import re
from textstat.textstat import textstatistics
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
  

def pos_score(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    return score['pos']


def negative_score(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    return score['neg']


def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity
def getPolarity(text):
   return TextBlob(final_article).sentiment.polarity

def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words
 
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)


def syllables_count(word):
    return textstatistics().syllable_count(word)


def break_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return list(doc.sents)


def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length


def complex_words(text):
     
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
    complex_words_set = set()
     
    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            complex_words_set.add(word)
 
    return len(complex_words_set)
def Avg_num_of_words_per_sentence(text):
    line_count=0
    words= text.split()
    word_count=len(words)
    for line in text:
        line_count+=1
    average_words=word_count/ sentence_count(text)
    return average_words

def Avg_word_length(text):
    text=text.split()
    res = sum(map(len, text))/float(len(text))
    return res

def syllables_count(word):
    return textstatistics().syllable_count(word)

def percentage_complex_word(text):
    per_diff_words = ( complex_words(final_article)/ word_count(final_article) * 100)
    return per_diff_words

def fog_index(text):
    grade = 0.4 * (avg_sentence_length(text) + percentage_complex_word(text))
    return grade

def personal_pronous(text):
    pronounRegex = re.compile(r'I|we|my|ours|us',re.I)
    pronouns = pronounRegex.findall(text)
    result=len(pronouns)
    return result

def syllable_count_per_word():
    res=syllables_count(final_article)/word_count(final_article)
    return res

print("subjectivity score :",getSubjectivity(final_article))
print("polarity score :",getPolarity(final_article))
print("positive score :",pos_score(final_article))
print("negative score :",negative_score(final_article))
print("percentage of complex words",percentage_complex_word(final_article))
print("complex word count",(complex_words(final_article)))
print("word count",word_count(final_article))
print("fog index :",fog_index(final_article))
print("syllable count per word",syllable_count_per_word())
print("Average sentence length: ",avg_sentence_length(final_article))
print("count of personal Pronous",personal_pronous(final_article))
print("Avarage word count per sentence :",Avg_num_of_words_per_sentence(final_article))
print("Avarage word length:",Avg_word_length(final_article))


# In[ ]:




