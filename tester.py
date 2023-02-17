from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import pandas as pd
#Train
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
import re 
from keras.models import load_model

###############################################################################################################
app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Dokumentasi API untuk Data Processing dan Modeling')
        }, host = LazyString(lambda: request.host)
    )
swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
    }
swagger = Swagger(app, template=swagger_template, config=swagger_config)

##############################################################################################################
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

sentiment = ['positive', 'neutral', 'negative']
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

vocab_size = len(tokenizer.word_index)

X1 = open("x_pad_sequences.p",'rb')
X = pickle.load(X1)

dummy_file = open('dummy.p', 'rb')
dummy = pickle.load(dummy_file)

model1 = open('model_neural3.p', 'rb')
model_neural = pickle.load(model1)

count = open('feature.p', 'rb')
count_vect = pickle.load(count)

model = load_model('model3.h5')

##############################################################################################################
#POSTNeural
@swag_from("docs/neural_post.yml", methods=['POST'])
@app.route('/neural_post', methods=['POST'])

def addNeural():
    Tweet = request.json['Tweet']
    text_clean = cleansing(Tweet)

    text = count_vect.transform([cleansing(text_clean)])

    hasil = model_neural.predict(text)[0]
    polarity = np.argmax(hasil[0])

    json_response = {
        'status_code' : 200,
        'description' : 'predict using neural',
        'bad_text' : text_clean,
        'sentiment' : sentiment[polarity]
    }

    return jsonify(json_response)

##############################################################################################################
# POST LSTM
@swag_from("docs/lstm_post.yml", methods=['POST'])
@app.route('/lstm_post', methods=['POST'])
def addLSTM():
    Tweet = request.json['Tweet']
    text_clean = cleansing(Tweet)

    tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
    predicted = tokenizer.texts_to_sequences(text_clean)
    guess = pad_sequences(predicted, maxlen=X.shape[1])

    prediction = model.predict(guess)
    polarity = np.argmax(prediction[0])

    json_response = {
        'status_code' : 200,
        'description' : 'predict using LSTM',
        'bad_text' : text_clean,
        'sentiment' : sentiment[polarity]
    }

    print(polarity)

    return jsonify(json_response)


##############################################################################################################
# POST CNN
@swag_from("docs/cnn_post.yml", methods=['POST'])
@app.route('/cnn_post', methods=['POST'])

def addCNN():
    Tweet = request.json['Tweet']
    text_clean = cleansing(Tweet)

    sentiment = ['negative', 'neutral', 'positive']

    predicted = tokenizer.texts_to_sequences(text_clean)
    guess = pad_sequences(predicted, maxlen=X.shape[1])

    model = load_model('model_CNN3.h5')
    prediction = model.predict(guess)
    polarity = np.argmax(prediction[0])


    json_response = {
        'status_code' : 200,
        'description' : 'predict using CNN',
        'bad_text' : text_clean,
        'sentiment' : sentiment[polarity]
    }

    return jsonify(json_response)

##############################################################################################################
if __name__ == "__main__":
    app.run()