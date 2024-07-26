
import streamlit as st
import joblib

#Data wrangling
import pandas as pd
import datetime
import re

#NLP
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
import contractions
import spacy
nlp = spacy.load("en_core_web_sm")

#Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def clean_text(text):
  """
  This function cleans the text by removing special characters, contractions, some spelling errors and stop words.
  """
  text = re.sub('[^A-Za-z ]', '', text)
  text = contractions.fix(text)
  text = text.replace("svc", "service")
  text = ' '.join([word for word in text.split() if word.lower() not in (stop)])
  return text

def entity_extraction(text):
  """
  This function extracts entities by checking whether they are valid English words or not and has a length >1.
  """
  entities=""
  for word in text.split():
      if word.lower() not in words and len(word)>1:
          entities+=word[0].upper()+word[1:].lower()+" "
  return entities

def remove_entities(text):
  """
  This function removes the entities and non-English words from the text.
  """
  text = ' '.join([word.lower() for word in text.split() if word.lower() in words])
  return text


pipeline=joblib.load('pipeline.pkl')
classes=joblib.load('classes.pkl')

st.set_page_config(page_title="Transaction Classifier")

st.markdown("# Transaction Classifier")

amount=st.number_input('Amount')
date=st.date_input('Date')
time = st.time_input('Time')
description=st.text_input('Description')
investment=st.checkbox('Interested in Investment')
credit=st.checkbox('Interested in Credit Building')
income=st.checkbox('Interested in Increasing Income')
payoff=st.checkbox('Interested in Paying Off Debt')
manage_spending=st.checkbox('Interested in Manage Spending')
submitted=st.button("Submit")

if submitted:
  df=pd.DataFrame({'amount':amount,'txn_date':date,'txn_time':time,'description':description, \
                   'IS_INTERESTED_INVESTMENT':investment,'IS_INTERESTED_BUILD_CREDIT':credit, \
                   'IS_INTERESTED_INCREASE_INCOME': income, 'IS_INTERESTED_PAY_OFF_DEBT': payoff, \
                   'IS_INTERESTED_MANAGE_SPENDING':manage_spending},index=[0])
  df['txn_date']=pd.to_datetime(df['txn_date'].astype(str) + " "+ df['txn_time'].astype(str))
  df.drop('txn_time',axis=1,inplace=True)
  df['dow']=df['txn_date'].dt.day_name()
  df['month']=df['txn_date'].dt.month.astype(int)
  df['day']=df['txn_date'].dt.day.astype(int)
  df['hour']=df['txn_date'].dt.hour.astype(int)
  df['minute']=df['txn_date'].dt.minute.astype(int)
  df['second']=df['txn_date'].dt.second.astype(int)
  df.drop("txn_date",axis=1,inplace=True)
  df['is_payment']=df['amount']<0
  df['amount']=df['amount'].abs()
  df['cleaned_description'] = df['description'].apply(clean_text)
  df['entities'] = df['cleaned_description'].apply(entity_extraction)
  df['cleaned_description'] = df['cleaned_description'].apply(remove_entities)
  df['keywords']=df['cleaned_description']+" "+df['entities']
  df['keywords']=df['keywords'].str.lower()
  df.drop(["description","cleaned_description","entities"],axis=1,inplace=True)
  res=pipeline.predict(df)[0]
  st.write(f"Predicted Category: {classes[res]}")
