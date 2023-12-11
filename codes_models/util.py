from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
import pandas as pd
import numpy as np

# Preprocessing
def preprocess(sentence):
    # lowercase
    sentence = sentence.lower()

    # remove punctuations
    sentence = "".join([char for char in sentence if char not in string.punctuation])

    # remove numbers
    sentence = re.sub(r'\d+', '', sentence)

    words = sentence.split(" ")

    # remove stopwords
    stop_words = stopwords.words("english")
    words = [word for word in words if word not in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]


    return " ".join(words)


def get_data(train_dir=train_dir, test_dir=test_dir, fn=fn, seeds=seeds, seed_idx=0):
    # load data
    train   = pd.read_excel(f"{train_dir}/{fn}-train-{seeds[seed_idx]}.xlsx")
    test    = pd.read_excel(f"{test_dir}/{fn}-test-{seeds[seed_idx]}.xlsx")

    # preprocess the textual data
    train["sentence"]   = train["sentence"].apply(preprocess)
    test["sentence"]    = test["sentence"].apply(preprocess)

    return train["sentence"], test["sentence"], train["label"], test["label"]



def get_sentence_embedding_glove(sentence:str,glove_vectors)->np.ndarray:

    vector_size = glove_vectors.vector_size
    sentence_emd = np.zeros(vector_size)
    words = sentence.split()

    for word in words:
        if word in glove_vectors:
            # add the word vector if its embedding is present in w2v
            sentence_emd += glove_vectors[word]

    if len(words) > 0:
        # compute the element-wise average of the word embeddings
        sentence_emd = sentence_emd / len(words)

    return sentence_emd
