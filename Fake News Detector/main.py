import pandas as pd
import numpy as np
import Predictor as fnp
import PySimpleGUI as sg


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

fnpred = fnp.FN_Predictor('data/fake_or_real_news.csv')
s = fnpred.accuracy()

# All the stuff inside your window.
layout = [  [sg.Text("Copy the article below")],
            [sg.InputText()],
            [sg.Button('Ok'), sg.Button('Score'), sg.Button('Cancel')] ]

# Create the Window
window = sg.Window('Hello Example', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()

    # if user closes window or clicks cancel
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    if event == 'Score':
        print(s)

    text = values[0]
    val = fnpred.predict(text)

    if (val == 0) & (event == 'Ok'):
        print('True')
    elif event == 'Ok':
        print('False')

window.close()



