#!/bin/bash

pip3 install gensim
pip3 install IPython
pip3 install matplotlib
pip3 install numpy
pip3 install pandas
pip3 install pickle
pip3 install pyldavis
pip3 install nltk
pip3 install seaborn
pip3 install scipy
pip3 install spacy
pip3 install sklearn
pip3 install tqdm

spacy -m downlaod en

python3 ../src/topic_modeling.py
