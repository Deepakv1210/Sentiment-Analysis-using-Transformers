import streamlit as st
from helper import *
st.title("Yelp Review Classifier")
st.button("Create Databese")
q=st.text_input("Review: ")
if q:
    #print(q)
    sentiment = predict_sentiment(model, tokenizer, q, max_len=128)
    st.header("User Sentiment: ")
    st.write(sentiment)