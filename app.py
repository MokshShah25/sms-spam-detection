from pyexpat import model
from codecs import lookup_error
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import time

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalpha():
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@st.cache_resource
def load_models():
    with open("model.pkl","rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl","rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return model,vectorizer

model,vectorizer = load_models()

st.title("SMS Spam Detection")

st.write("Enter a message below to check whether it is Spam or Not Spam.")

user_input = st.text_input("Enter SMS here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message first")
    else:
        with st.spinner("Processing your message..."):
            time.sleep(0.5)
            transformed_sms = transform_text(user_input)
            transformed_input = vectorizer.transform([transformed_sms])
            prediction = model.predict(transformed_input)
            
            if prediction[0] == 1:
                st.error("Spam")
            else:
                st.success("Not Spam")  