import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [token for token in text if token.isalnum()]
    text = [token for token in text if token not in list(stopwords.words('english'))]
    text = [token for token in text if token not in string.punctuation]
    ps = PorterStemmer()
    text = [ps.stem(token)for token in text]
    text = ' '.join(text)
    return text

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl,'rb')

st.title("AI Jailbreak Prompt Classifier")

input_prompt = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_text_ai = transform_text(input_prompt)
    # 2. vectorize
    vectorized = vectorizer.transform(transformed_text_ai).toarray()
    # 3. predict
    result = model.predict(vectorized)[0]
    # 4. Display
    if result == 1:
        st.header(f"Jailbreak")
    else:
        st.header(f"Benign")
