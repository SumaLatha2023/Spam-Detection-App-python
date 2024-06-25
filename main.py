import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("dataset.csv")
# print(data.head())

# print(data.shape)
data.drop_duplicates(inplace=True)
# print(data.shape)

data.isnull().sum()

data['Category'] = data['Category'].replace(['ham','spam'],['NOT SPAM','SPAM'])
# print(data.head())

cat = data['Category']
msg = data['Message']

(msg_train,msg_test,cat_train,cat_test) = train_test_split(msg, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(msg_train)

# creating model
model = MultinomialNB()
model.fit(features, cat_train)

# testing model
features_test = cv.transform(msg_test)
model.score(features_test, cat_test)

# predict data
def predict(message):
  input_message = cv.transform([message]).toarray()
  result = model.predict(input_message)
  return result

st.header('Email/SMS Spam Detector')

input = st.text_input('Enter message here')

if st.button('Predict'):
  output = predict(input)
  st.markdown(output)
