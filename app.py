from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import time
import re
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost
from gensim.utils import simple_preprocess
from flasgger import Swagger

app = Flask(__name__)
pickle_in1 = open("fasttext_model.pkl", "rb")
model_ft = pickle.load(pickle_in1)
pickle_in2 = open("fasttext_ctb.pkl", "rb")
ctb = pickle.load(pickle_in2)
pickle_in3 = open("model_w2v_4.pkl", "rb")
w2v = pickle.load(pickle_in3)
#pickle_in = open("w2v_model_ctb2.pkl", "rb")
#w2v_model_ctb = pickle.load(pickle_in)
w2v_model_ctb = CatBoostClassifier()
w2v_model_ctb.load_model('w2v_model_ctb_new')
pickle_in4 = open("bert_ctb.pkl", "rb")
bert_ctb = pickle.load(pickle_in4)
vocab = open('vocab.txt', 'r')
bert_config = open('bert_config.json', 'r')
bert_model = open('bert_model.ckpt.index', 'r')

def re_urls(text, replace_for="URL"):
    return re.sub(r'https?://\S+', replace_for, text)


def re_digits(text, replace_for='DIGIT'):
    result = re.sub(r'\d+', replace_for, text)
    return result


def preprocessing(doc):
    doc = re_urls(doc)
    doc = re_digits(doc)
    return doc


def prepare_input(text):
    input_series = pd.Series(text)
    input2 = input_series.map(preprocessing)
    input3 = input2.map(simple_preprocess)
    print(input3)
    return input3


def get_doc2vec_X(model, tokens):
    def __calc_doc2vec(words):  # vectorization
        return np.mean([model.wv[w] for w in words if w in model.wv], axis=0)

    X = tokens.map(__calc_doc2vec)
    print(X)
    default_vector = X[False == X.isnull()].mean()
    print(default_vector)
    #default_vector = -1
    print("8")
    return np.stack(X.map(lambda x: default_vector if str(x) == 'nan' else x))

def predict_tweet_word2vec(input):
    prep_input = prepare_input(input)
    print("3")
    print(prep_input)
    time.sleep(3)
    x_input = get_doc2vec_X(w2v, prep_input)
    time.sleep(5)
    print("4")
    print(x_input)
    pred = w2v_model_ctb.predict(x_input)
    time.sleep(3)
    print(pred)
    return pred

def predict_tweet_fasttext(input):
    prep_input = prepare_input(input)
    x_input = get_doc2vec_X(model_ft, prep_input)
    pred = ctb.predict(x_input)
    print(pred)
    return pred


@app.route('/')
def index():
    return render_template('login.html')


@app.route("/", methods=['GET', 'POST'])
def perform_prediction():
    input = request.form['input']
    embedding = request.form["emb"]
    your_script_result = 'The output is'

    if embedding == "w2v":
        output = predict_tweet_word2vec(input)
    elif embedding == "fasttext":
        output = predict_tweet_fasttext(input)
    elif embedding == "bert":
        output = predict_tweet_fasttext(input)
    else:
        output = "Please choose embedding form first"

    if output[0] == 1:
        output_str = "This is disaster tweet"
    else:
        output_str = "This is not disaster tweet"

    return render_template('index.html', input_form=output_str, embedding_form=embedding)


if __name__ == '__main__':
    app.run()
