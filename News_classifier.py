import streamlit as st
import pickle
import nltk

from nltk.stem import  PorterStemmer
from nltk.corpus import stopwords
import string 
ps=PorterStemmer()
def preprocess_news(data):
    data=data.lower()
    data=nltk.word_tokenize(data)
    y= []
    
    for i in data:
        if i.isalnum():
            y.append(i)
    data= y[:]
    y.clear()
    for i in data:
        if i not in stopwords.words('english') and i not  in string.punctuation:
            y.append(i)
            
    data= y[:]
    y.clear()
    for i in data:
        
        y.append(ps.stem(i))
        
    return " ".join(y)
model=pickle.load(open(r"C:\Users\bisht\OneDrive\Desktop\News_classifier\model.pkl",'rb'))
cv=pickle.load(open(r"C:\Users\bisht\OneDrive\Desktop\News_classifier\vectorizer.pkl",'rb'))
st.title("News Classfier")
input_news=st.text_input("Enter The News")
if st.button('Predict'):
    preprocess_news=preprocess_news(input_news)
    vector_input=cv.transform([preprocess_news])
    result=model.predict(vector_input)[0]

    if result==0:
        st.header('Political')
    elif result==1:
        st.header('Technology')
    elif result==2:
        st.header('Entertainment')
    else:
        st.header('bussiness')