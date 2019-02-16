# for text handling
import string
import nltk
from nltk import tokenize
from nltk import corpus
# for distance calculation
import numpy as np
from math import acos, sqrt
# for command line arguments handling
import sys
# for stemming and lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

default_lemmatizer = WordNetLemmatizer()
default_stemmer = PorterStemmer()

# remove characters occuring in @removals from another string
def remove_chars(s, removals):
    return s.translate(str.maketrans('', '', removals))

# split text into word lists
def tokenize(text, lmtzr = default_lemmatizer, stmr = default_stemmer):
    # remove punctuation
    text = remove_chars(text, string.punctuation)
    # remove numbers and stem words
    # words = [word.lower() for word in text.split() if word.isalpha()]
    # words = [lmtzr.lemmatize(word.lower()) for word in text.split() if word.isalpha()]
    words = [stmr.stem(word.lower()) for word in text.split() ] #if word.isalpha()]
    # remove stopwords, but remove punctuation from the stopwords first because we removed all punctuation from the original text too
    stopwords = set([remove_chars(sw, string.punctuation) for sw in corpus.stopwords.words('English')])
    words = [word for word in words if word not in stopwords]
    return words

# returns a count vector for a list of words and a vocabulary
def count_vectorize(words, vocabulary):    
    # compute frequencies and vocabulary for list of words
    fdist = nltk.FreqDist(words)
    freqs = np.array([freq for (word, freq) in fdist.items()])
    # get words that are in the vocabulary but not in the current list of words
    words_not_present = [word for word in vocabulary if word not in fdist.keys()]
    for word in words_not_present:
        fdist[word] = 0    
    # get word frequencies in the count vector
    v = np.array([fdist[word] for word in vocabulary])
    return v

#compute distance according to formula
# !!! arccos((V1*V2)/sqrt(V1*V1)*(V2*V2))
def compute_distance(v1, v2):
    distance = acos( (np.dot(v1,v2) / sqrt( (np.dot(v1,v1) * np.dot(v2,v2)))))
    return distance


def compute_similarity(text1, text2):
    w1, w2 = tokenize(text1), tokenize(text2)
    # create a sorted common vocabulary for the two lists of words
    common_vocab = sorted(set(w1) | set(w2))

    # create count vectors of the same lenght for the two lists of words
    v1 = count_vectorize(w1, common_vocab)
    v2 = count_vectorize(w2, common_vocab)

    distance = compute_distance(v1, v2)
    return distance

if __name__ == '__main__':

    if len(sys.argv) != 3 :
        print("Two text files expected as arguments!")
        print("Script will compute similarity for them !")
        sys.exit(1)

    texts = []
    for filename in sys.argv[1:] :
        with open(filename, 'r') as file:
            texts.append(file.read())

    text1, text2 = texts[0], texts[1]

    distance = compute_similarity(text1, text2)
    print('Distanta intre cele doua texte este {:.4f}.'.format(distance))

