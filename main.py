#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:20:35 2019

@author: njemoui
"""

! pip3 install pymongo
! pip3 install fasttext
! pip3 install pandas
! pip3 install matplotlib
! pip3 install seaborn
! pip3 install sklearn
! pip3 install tensorflow-gpu
! pip3 install keras
! pip3 install spacy
! pip3 install nltk
! python3 -m spacy download en_core_web_lg


import xml.etree.ElementTree as ET
import csv, os

! wget https://nlp.stanford.edu/projects/contradiction/RTE1_dev1_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE1_dev2_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE1_test_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE2_dev_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE2_test_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE3_dev_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE3_test_3ways.xml
! mkdir ./data
! mv ./*.xml ./data/


xml_files = [file for file in os.listdir(os.getcwd()+"/data") if file.endswith(".xml")]
xml_files

import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
nlp = spacy.load('en_core_web_lg')

csv_file = open("./data/" + "total_dataset" + ".csv",'w')
csv_writer = csv.writer(csv_file,delimiter='|')
csv_writer.writerow(["entailment","task","t","h","t_token","t_lemm","t_stem","t_pos","t_ents","h_token","h_lemm","h_stem","h_pos","h_ents"])


def natural_language_processing(text):
            doc = nlp(text)
            t_token = []
            t_lemm = []
            t_stem = []
            t_pos = []
            t_ents = []
            for token in doc:
                t_token.append(token.text)
                t_lemm.append(token.lemma_)
                t_pos.append(token.pos_)
                t_stem.append(ps.stem(token.text))
            for ent in doc.ents:
                t_ents.append([ent.text, ent.start_char, ent.end_char, ent.label_])
            return t_token, t_lemm, t_stem, t_pos, t_ents
        
for file in xml_files:
    tree = ET.parse("./data/" + file)
    root = tree.getroot()
    pairs = root.findall('pair')
    for pair in pairs:
        row = []
        row.append((pair.attrib)['entailment'])
        row.append((pair.attrib)['task'])
        row.append(pair.find('t').text)
        row.append(pair.find('h').text)
        t_token, t_lemm, t_stem, t_pos, t_ents = natural_language_processing(pair.find('t').text)
        row.append(t_token)
        row.append(t_lemm)
        row.append(t_stem)
        row.append(t_pos)
        row.append(t_ents)
        h_token, h_lemm, h_stem, h_pos, h_ents = natural_language_processing(pair.find('h').text)
        row.append(h_token)
        row.append(h_lemm)
        row.append(h_stem)
        row.append(h_pos)
        row.append(h_ents)
        csv_writer.writerow(row)
        

        
import pandas as pd
dataset = pd.read_csv('./data/total_dataset.csv', delimiter='|', error_bad_lines=False,engine = 'python')

dataset.head(10)


import fasttext as ft
def prepare_fasttext_model(df):
        file = open("data/fasttext_model_sentence_per_sentence.txt", "w")
        for t,h in zip(df['t'].values,df['h'].values):
            file.write(t + "\n")
            file.write(h + "\n")
        file.close()
        ft.cbow("data/fasttext_model_sentence_per_sentence.txt", "data/fasttext_model_sentence_per_sentence")
prepare_fasttext_model(dataset)



dataset["combined"] = dataset[['t_lemm', 'h_lemm']].apply(lambda line: ' '.join([" ".join(line[0].replace('[','').replace(']','').split(','))," ".join(line[1].replace('[','').replace(']','').split(','))]), axis=1)
dataset["combined"] = dataset["combined"].apply(lambda line: line.replace("'",""))
dataset["entailment"] = dataset["entailment"].apply(lambda label: 1 if label == "YES" else 0)
dataset.head()

from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(dataset['combined'], dataset['entailment'])

import numpy as np
from keras.preprocessing import text, sequence
def making_embeddings_and_sequences(train_df,train_x,valid_x):
    embeddings_index = {}
    for i, line in enumerate(open('data/fasttext_model_only_text.vec')):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    token = text.Tokenizer()
    token.fit_on_texts(train_df['combined'])
    word_index = token.word_index
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=800)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=800)
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return train_seq_x, valid_seq_x, word_index, embedding_matrix, token

train_seq_x, valid_seq_x, word_index, embedding_matrix, token = making_embeddings_and_sequences(dataset,train_x, valid_x)

# %timeit -n number_of_loops routine
dataset.combined.map(len).max()

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import model_from_json
from sklearn import model_selection
from keras.optimizers import Adam
from keras.layers import Bidirectional
import tensorflow as tf


def create_the_model(len_word_index,embedding_matrix):
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Sequential()
        model.add(Embedding(len_word_index + 1, 100, weights=[embedding_matrix], trainable=False))
        model.add(Bidirectional(LSTM(units=100)))
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])
        return model
    
def configure_the_gpu():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
        
configure_the_gpu()

model = create_the_model(len(word_index),embedding_matrix)


model.fit(train_seq_x, train_y, epochs=100, batch_size=128)












































