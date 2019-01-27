#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:16:20 2019

@author: njemoui
"""


! wget https://nlp.stanford.edu/projects/contradiction/RTE1_dev1_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE1_dev2_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE1_test_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE2_dev_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE2_test_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE3_dev_3ways.xml
! wget https://nlp.stanford.edu/projects/contradiction/RTE3_test_3ways.xml
! mkdir ./data
! mv ./*.xml ./data/

import xml.etree.ElementTree as ET
import csv, os

xml_files = [file for file in os.listdir(os.getcwd()+"/data") if file.endswith(".xml")]
xml_files

import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
ps = PorterStemmer()
nlp = spacy.load('en_core_web_lg')

ent_l = []
tag_l = []
dep_l = []

def natural_language_processing(text):
    doc = nlp(pair.find('t').text)
    ents = {}
    dictionnaire_list = []
    for ent in doc.ents:
        ents[str(ent.start_char) + "," + str(ent.end_char)] = ent.label_
    for token in doc:
        dictionnaire = {
            "TEXT" : None,
            "STEM" : None,
            "LEMMA" : None,
            "TAG" : None,
            "DEP" : None,
            "ALPHA" : None,
            "ENTITIE" : None,
            }
        for key in ents:
            range_ = key.split(',')
            if (doc.text).find(token.text) in range(int(range_[0]),int(range_[1])):
                dictionnaire["ENTITIE"] = ents[key]
                ent_l.append(ents[key])
        dictionnaire["TEXT"] = token.text
        dictionnaire["LEMMA"] = token.lemma_
        dictionnaire["STEM"] = ps.stem(token.text)
        dictionnaire["TAG"] = token.tag_
        dictionnaire["ALPHA"] = token.is_alpha
        dictionnaire["DEP"] = token.dep_
        
        tag_l.append(token.tag_)
        dep_l.append(token.dep_)
        dictionnaire_list.append(dictionnaire)
        
    
    return dictionnaire_list
final_list = []
for file in xml_files:
    tree = ET.parse("./data/" + file)
    root = tree.getroot()
    pairs = root.findall('pair')
    for pair in pairs:
        dictionnaire = {}
        text = pair.find('t').text
        dictionnaire["text"] = text
        dictionnaire["basic_features"] = natural_language_processing(text)
        dictionnaire = {}
        text = pair.find('h').text
        dictionnaire["text"] = text
        dictionnaire["basic_features"] = natural_language_processing(text)
        final_list.append(dictionnaire)
        (exit(0))
with open('./data/final_list.pkl', 'wb') as output:
    pickle.dump(final_list, output, pickle.HIGHEST_PROTOCOL)
