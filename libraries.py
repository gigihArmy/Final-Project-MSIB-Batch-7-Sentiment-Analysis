#Library
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import contractions
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC