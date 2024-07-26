import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('WELFake_Dataset.csv')
news_df.dropna(inplace=True)
X = news_df.drop('label',axis=1)
Y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]',' ',title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [ps.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = ' '.join(stemmed_title)
    return stemmed_title

# Apply stemming function to content column
news_df['title'] = news_df['title'].apply(stemming)

# Vectorize data
X = news_df['title'].values
Y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)


# website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')