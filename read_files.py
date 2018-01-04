#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding=utf8

import os
import re
import random
import nltk
from nltk import tokenize
import string
import numpy as np

DATA_FOLDER_BLOG = "CourseraSwift"

DATA_FOLDER_PLAIN = "books"

num_vocabs = 10000

TEXT_DATA_SIZE = 0

MERGE_DATA = True

PROCESSED_VANILLA_TEXT = "plain.txt"

PROCESSED_BLOG_TEXT = "blog.txt"


def get_processed_text(data_folder_name, cut_off=False):
    text = ''
    for filename in sorted(os.listdir(data_folder_name)):
        path = data_folder_name + '/' + filename
        line_text = ''
        with open(path, 'r') as f:
            line_text += str(f.read().lower())
            line_text = ''.join(
                filter(lambda x: x in string.printable, line_text))
            line_text = line_text.replace("“", '"').replace("”", '"').replace(
                "’", "'")
            line_text = line_text.replace('\r\n', '\n')
            line_text = line_text.replace("“", '"').replace("”", '"').replace(
                "’", "'")
            line_text = line_text.translate(None, string.punctuation)

        text += line_text

        print "File {0} completed".format(path)

    if cut_off:
        text = text[:TEXT_DATA_SIZE]

    print "Length of File:", len(text)
    return text


vanilla_text = ''
blog_text = ''

try:
    print("Trying to read from exisiting files")
    with open(PROCESSED_VANILLA_TEXT, 'r') as f:
        vanilla_text = f.read()
    with open(PROCESSED_BLOG_TEXT, 'r') as f:
        blog_text = f.read()
    print("Successfully read")
except:
    print("Failed to read processed text. Reading new data")
    vanilla_text = get_processed_text(DATA_FOLDER_PLAIN)
    with open(PROCESSED_VANILLA_TEXT, 'w') as f:
        f.write(vanilla_text)

    TEXT_DATA_SIZE = len(vanilla_text)
    blog_text = get_processed_text(DATA_FOLDER_BLOG, cut_off=True)

    with open(PROCESSED_BLOG_TEXT, 'w') as f:
        f.write(blog_text)

if MERGE_DATA:
    print ("Merging the data from both files")
    text = vanilla_text + blog_text
else:
    text = vanilla_text

print "Length of File After Merge:", len(text)

paras = re.split(r"\n+", text)

random.seed(500)
random.shuffle(paras)
text = '\n\n'.join(paras)
tokens = tokenize.word_tokenize(text)

print "Tokens Generated"

frequent_tokens = nltk.FreqDist(tokens)
token_counts = frequent_tokens.most_common(num_vocabs - 1)

all_tokens = [token_count[0] for token_count in token_counts]

all_tokens.insert(0, '')
token_with_index_dict = dict([(token, i) for i, token in enumerate(all_tokens)])

print "Processing Tokens"

sequence = [token_with_index_dict.get(token) for token in tokens if
            token_with_index_dict.get(token, 0) != 0]

num_of_elements = len(sequence)

sequence = np.array(sequence, dtype=np.int)

indx_to_word = {indx: word for indx, word in enumerate(all_tokens)}

print "Tokens processed"

word_vectors = {}
with open('glove.6B/glove.6B.100d.txt', 'r') as f:
    for line in f:
        data = line.split()
        word = data[0]
        coefs = np.asarray(data[1:], dtype='float32')
        word_vectors[word] = coefs


def split_data(n, number_of_testing, train_amount=1.0):
    total_num_training = num_of_elements - number_of_testing
    num_of_trainig = int(total_num_training * train_amount)

    print "Creating train and test sets"

    x_train, y_train = generate_dataset(sequence, n=n, noffset=0,
                                        nelements=num_of_trainig)
    x_test, y_test = generate_dataset(sequence, n=n, noffset=num_of_trainig,
                                      nelements=number_of_testing)

    return x_train, y_train, x_test, y_test


def generate_dataset(sequence, n, offset, nelements):
    xs, ys = [], []
    for _ in range(offset, offset + nelements - n - 1):
        xs.append(sequence[i:i + n - 1])
        ys.append(sequence[i + n - 1])
    return np.array(xs), np.array(ys)


def get_processed_data():
    global word_vectors
    global token_with_index_dict
    global indx_to_word
    return word_vectors, token_with_index_dict, indx_to_word
