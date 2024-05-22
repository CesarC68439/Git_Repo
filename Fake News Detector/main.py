import pandas as pd
import numpy as np
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv('data/fake_or_real_news.csv')
display(data)