# -*- coding: utf-8 -*-
"""Fake_News_Prediction (1).ipy
e Name:- Rushikesh Ashokrao Girhe

#Subject:- Machine Learning Internship at RemarkSkill

#Project Name:- Fake News Prediction

**Steps to solve the Project**

1. In this project we are going to see how we can build a machine learning system that can predict whether a news is fake or real okay so this is a very interesting project because in this case we are going to use textual data 


 2. let's understand the workflow we are going to follow first is the collection of data so we need to collect this news data

3. The data set we are going to use is basically a label data set so it consists of several thousand of news articles and it will be labeled as either it is real news or fake news.

4. It also contains other details such as the author of that particular news the title of the news etc okay so once we have this data set we need to pre-process this data 

5. we train our machine learning model so we need to evaluate our model  so that can be done using the test data 

6. In this project we are going to use a logistic regression model because this is a binary classification means we are going to classify the result into two types it's either real or fake.

7. So once we train this logistic regression model we get a trained model so we will do some evaluations on this model. we find the accuracy score of this model using the test data now once that done we have a trained logistic regression model 

8. In this trained regression model so we feed new data to this model so for those which we don't know whether the news is real or fake. once we give the data to our model it can predict whether the news is real or fake

About the Dataset:

1. id: unique id for a news article
2. title: the title of a news article
3. author: author of the news article
4. text: the text of the article; could be incomplete
5. label: a label that marks whether the news article is real or fake:
           1: Fake news
           0: real News

Importing the Dependencies
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st 
import pickle


st.title('Fake News Prediction Using Machine Learning')
st.subheader('Devloped by Rushikesh Girhe Under the guidence of nirde sir')
st.text_input("Enter news you want to predict")
st.image("identify_fn.jpeg")


import nltk
nltk.download('stopwords')

# printing the stopwords in English
#print(stopwords.words('english'))

news_dataset = pd.read_csv(r"C:\Users\Rushikesh\Documents\Remark_Skill_ML_Internship\Fake news prediction\train.csv")
#print(news_dataset)

#Data Pre-processing

news_dataset.shape

# print the first 5 rows of the dataframe
news_dataset.head()

# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

#print(news_dataset['content'])

# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

#print(X)
#print(Y)

#Stemming:Stemming is the process of reducing a word to its Root word

#example:actor, actress, acting --> act


port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

#print(news_dataset['content'])

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

#print(X)

#print(Y)

Y.shape

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

#print(X)

#Splitting the dataset to training & test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

#Training the Model: Logistic Regression

model = LogisticRegression()

model.fit(X_train, Y_train)

#Evaluation accuracy score

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#print('Accuracy score of the training data : ', training_data_accuracy)

st.write('Accuracy score of the training data : ')
st.write(training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#print('Accuracy score of the test data : ', test_data_accuracy)

st.write('Accuracy score of the testing data : ')
st.write(test_data_accuracy)

pickle.dump(model, open('training.pkl', 'wb'))


#Making a Predictive System

X_new = X_test[3]

#print(X_new)
st.dataframe(X_new)
#print()
prediction = model.predict(X_new)
#print(prediction)


if (prediction[0]==0):
  #print('The news is Real')
  output='The news is Real'
else:
  #print('The news is Fake')
  output='The news is Fake'

result=st.button('Predict result')
st.write(result)
if result:
  st.write(output)

#print(Y_test[3])
