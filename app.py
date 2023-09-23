#!/usr/bin/env python
# coding: utf-8

# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import string
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from numpy import dot
from numpy.linalg import norm

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Importing datasets
df = pd.read_csv('https://drive.google.com/u/0/uc?id=1R1w1K9gzfMKyDjG9SpoTu0KsC48Q1XP5&export=download')
df1 = pd.read_csv('https://drive.google.com/u/0/uc?id=1R1w1K9gzfMKyDjG9SpoTu0KsC48Q1XP5&export=download')

# Preprocessing data

# Function to remove all URLs from text
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text

# Make all text lowercase
def text_lowercase(text):
    return text.lower()

# Remove numbers from text
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# Remove punctuation from text
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Tokenize text
def tokenize(text):
    text = word_tokenize(text)
    return text

# Remove stopwords from text
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# Lemmatize words in text
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

# Function to preprocess text data
def preprocessing(text):
    text = text.replace('\n', ' ')  # Remove newline characters
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text

# Combine all preprocessing steps
def preprocessing_input(query):
    query = query.replace('\n', ' ')  # Remove newline characters
    query = preprocessing(query)
    K = get_mean_vector(word2vec_model, query)
    return K

# Load pre-trained word embedding models
skipgram = Word2Vec.load('skipgramx11.bin')
FastText = Word2Vec.load('fasttext.bin')

# Define vector size for each word
vector_size = 100

# Function to get the mean vector for a list of words
def get_mean_vector(word2vec_model, words):
    words = [word for word in tokenize(words) if word in list(word2vec_model.wv.index_to_key)]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.array([0] * 100)

# Load vectors from CSV files
K = pd.read_csv('skipgram-vec.csv')
K2 = [K[str(i)].values for i in range(df.shape[0])]

KK = pd.read_csv('fasttext-vec.csv')
K1 = [KK[str(i)].values for i in range(df.shape[0])]

# Function to calculate cosine similarity
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Set display option to show full text
pd.set_option("display.max_colwidth", -1)

# Streamlit function
def main():
    # Load data and models
    data = df1

    st.title("Clinical Trial Search Engine")
    st.write('Select Model')

    Vectors = st.selectbox("Model", options=['Skipgram', 'Fasttext'])
    if Vectors == 'Skipgram':
        K = K2
        word2vec_model = skipgram
    elif Vectors == 'Fasttext':
        K = K1
        word2vec_model = FastText

    st.write('Type your query here')

    query = st.text_input("Search box")

    # Preprocess user input query
    def preprocessing_input(query):
        query = query.replace('\n', ' ')  # Remove newline characters
        query = preprocessing(query)
        K = get_mean_vector(word2vec_model, query)
        return K

    # Function to find top N results based on cosine similarity
    def top_n(query, p, df1):
        query = preprocessing_input(query)
        x = []

        for i in range(len(p)):
            x.append(cos_sim(query, p[i]))
        tmp = list(x)
        res = sorted(range(len(x)), key=lambda sub: x[sub])[-10:]
        sim = [tmp[i] for i in reversed(res)]

        L = []
        for i in reversed(res):
            L.append(i)
        return df1.iloc[L, [1, 2, 5, 6]], sim

    model = top_n
    if query:
        P, sim = model(str(query), K, data)
        # Create a Plotly table to display results
        fig = go.Figure(data=[go.Table(header=dict(values=['ID', 'Title', 'Abstract', 'Publication Date', 'Score']),
                                       cells=dict(values=[list(P['Trial ID'].values), list(P['Title'].values),
                                                          list(P['Abstract'].values), list(P['Publication date'].values),
                                                          list(np.around(sim, 4))], align=['center', 'right']))])
        fig.update_layout(height=1700, width=700, margin=dict(l=0, r=10, t=20, b=20))
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
