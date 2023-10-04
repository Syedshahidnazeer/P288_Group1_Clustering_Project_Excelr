# Import flask and other libraries
import os

import pickle
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize

model_logistic = pickle.load(open("models/logistic_regression.sav", "rb"))

count_vectorizer = pickle.load(open("models/variables/count_vectorizer_var.sav", "rb"))

#functions
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(text):

    sentence = ""
    
    # Tokenization
    words = nltk.word_tokenize(text)
    # Stopwords removal
    words = [str(lemmatizer.lemmatize(w)).lower() for w in words if not w in stop_words]
    # Lemmatization
    for word in words:
        sentence = sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        
    return [sentence]

def vectorize_sentence(sentence):
    freq_term_matrix = count_vectorizer.transform(sentence)

    tfidf = TfidfTransformer(norm = "l2")
    tfidf.fit(freq_term_matrix)
    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
    
    return tf_idf_matrix

# create instance of Flask app
app = Flask(__name__)

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


# Define what to do when a user hits the /visuals route
@app.route("/visuals")
def visuals():
    return render_template("visuals.html")

@app.route("/form")
def form():
    return render_template("form.html", model_result = " ")

@app.route("/conclusion")
def conclusion():
    return render_template("conclusion.html")

@app.route("/aboutme")
def aboutme():
    return render_template("aboutme.html")
@app.route("/ppt")
def ppt():
    return render_template("ppt.html")
# Define what to do when a user hits the /model route
@app.route("/send",  methods=['POST', 'GET'])
def send():
    #print('hello')
    new_text = request.form["predictfakenews"]
    print(new_text)
    
    lemm_text = lemmatize_sentence(new_text)

    vect_text = vectorize_sentence(lemm_text)
    
    print(vect_text)
    #vectorize_sentence
    result = model_logistic.predict(vect_text)
    
    if result[0] == 1:
        result = "Real"
    else:
        result = "Fake"
    
    return render_template("result.html", model_result = result, new_text = new_text) 
  

# Run app
if __name__ == "__main__":
    app.run(debug=True)