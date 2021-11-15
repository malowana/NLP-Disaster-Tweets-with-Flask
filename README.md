# NLP-Disaster-Tweets-with-Flask

The aim of this project was to create a webpage, where user can test different embeddings form and NLP models.<br>
I create 3 models and use embeddings like: word2vec, FastText and BERT.<br>
Models were trained on dataset from Kaggle competition. <a href="https://www.kaggle.com/c/nlp-getting-started/data">LINK</a><br>
For BERT pretraining I used also 'bert-base-uncased' model. <a href="https://github.com/google-research/bert">LINK</a>
<br>
<br>
At the beginning provided text was cleaned up: digits and urls were replaced by text: DIGIT and URL. <br>
After that I use function simple_preprocess, which remove interpunction, convert letters to the small one etc. <br>
In next step this string was tokenized and convert to vector, based on previously choosen embedding. <br>
At the end we receive an output from catboost model. <br>
<br>
![This is an image](images/dt.PNG) <br><br>
[[Watch the video how it works]](https://www.youtube.com/watch?v=wYNPKVg1rhE)
