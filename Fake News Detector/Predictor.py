import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

class FN_Predictor:
    global classifier
    global X_test_vectorized
    global Y_test

    def __init__(self, string):
        data = pd.read_csv(string)
        data['fake'] = data['label'].apply(lambda x : 0 if x == 'REAL' else 1)

        x, y = data['text'], data['fake']
        X_train, X_test, Y_train, self.Y_test = train_test_split(x, y, test_size=0.2)

        self.vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.7)
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.X_test_vectorized = self.vectorizer.transform(X_test)

        clf = LinearSVC()
        clf.fit(X_train_vectorized, Y_train)
        self.classifier = clf
    
    def accuracy(self):
        score = self.classifier.score(self.X_test_vectorized, self.Y_test)
        return score
    
    def predict(self, string):
        vectorized_text = self.vectorizer.transform([string])
        arr = self.classifier.predict(vectorized_text)
        return arr[0]
    
        