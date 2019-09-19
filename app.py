from flask import Flask, render_template,url_for,request
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
import tensorflow
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pickle as p
import json
MAX_SEQUENCE_LENGTH=250
import socket
print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))
app = Flask(__name__)


@app.route('/')
#Takes input through form
def index():
    return render_template('index.html')
@app.route('/tensorinput',methods = ['GET','POST'])
#Takes input through form
def tensorinput():
    return render_template('tensorinput.html')


@app.route('/model',methods = ['GET','POST'])
def model():
    
    
    #have pickled my NB model as final.pickle
    clf = p.load(open('final.pickle','rb'))
    
    #data=request.get_json()
    #loading my pickled pipeline
    pipeline = joblib.load('pipeline.pickle')
    #getting data from form
    if request.method == 'POST':
    	doc1 = request.form['doc1']
    	#print(doc2)
    	doc2= request.form['doc2']
    	doc3= request.form['doc3']
    	docs=[]
    	docs.append(doc1)
    	docs.append(doc2)
    	docs.append(doc3)
    	#final docs stored
    	print(docs)
    	#passing input for model
    y_test = clf.predict_proba(pipeline.transform(docs))
    y_test=y_test[:,1]
    #print(y_test)
    return render_template('result.html', prediction = y_test)
@app.route('/tensormodel',methods = ['GET','POST'])
def tensormodel():
    
    preprocess = p.load(open('preprocess.pickle','rb'))
    #have loaded my tf model as my_tensorflow_model
    tf_model = keras.models.load_model('my_tensorflow_model.h5')
    
   
    #loading my pickled tokenizer
    tokenizer = joblib.load('tokenizer.pickle')
    #getting data from form
    if request.method == 'POST':
    	doc1 = request.form['doc1']
    	#print(doc2)
    	doc2= request.form['doc2']
    	doc3= request.form['doc3']
    	docs=[]
    	docs.append(doc1)
    	docs.append(doc2)
    	docs.append(doc3)
    	#final docs stored
    	print(docs)
    
    #preprocess Data and tokenize input

    seq=tokenizer.texts_to_sequences(docs)
    X_test=keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = tf_model.predict_proba(X_test)
    print(y_test)
    y_test=y_test[:,1]
    
    return render_template('result.html', prediction = y_test)

if __name__ == '__main__':
    app.run(debug=True)