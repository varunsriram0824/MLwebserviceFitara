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
import sqlite3




MAX_SEQUENCE_LENGTH=250

app = Flask(__name__)


@app.route('/')

#Takes us to main page
def index():
    return render_template('index.html')

@app.route('/bayesinput')

#Takes input through form
def bayesinput():
    return render_template('bayesinput.html')

@app.route('/tensorinput',methods = ['GET','POST'])

#Takes input through form
def tensorinput():
    return render_template('tensorinput.html')


@app.route('/model',methods = ['GET','POST'])

#Naive Bayes Model
def model():
 
    #have pickled my NB model as final.pickle
    clf = p.load(open('final.pickle','rb'))
    
    
    #loading my pickled pipeline
    pipeline = joblib.load('pipeline.pickle')

    #getting data from form
    if request.method == 'POST':
    	doc1 = request.form['comment1']
    	doc2= request.form['comment2']
    	doc3= request.form['comment3']
    	docs=[]
    	docs.append(doc1)
    	docs.append(doc2)
    	docs.append(doc3)

    	#final docs stored
    	print(docs)

    #passing input for model
    y_test = clf.predict_proba(pipeline.transform(docs))
    y_test=y_test[:,1]
    
    #Making Connection and Inserting Data
    conn = sqlite3.connect('fitara.db')
    if(conn):
    	print("Opened database successfully")

    modelname="Bayes"
    docname1=request.form['docname1']
    docname2=request.form['docname2']
    docname3=request.form['docname3']
    cur = conn.cursor()
    cur.execute("INSERT INTO results (Doc_name,Is_fitara,Model) VALUES (?,?,?)",(docname1,str(y_test[0]),modelname) )
    cur.execute("INSERT INTO results (Doc_name,Is_fitara,Model) VALUES (?,?,?)",(docname2,str(y_test[1]),modelname) )
    cur.execute("INSERT INTO results (Doc_name,Is_fitara,Model) VALUES (?,?,?)",(docname3,str(y_test[2]),modelname) )
    conn.commit()
    msg = "Record successfully added"
    conn.close()
    return render_template('status.html', msg = msg)

@app.route('/tensormodel',methods = ['GET','POST'])

#RNN Model
def tensormodel():
	conn = sqlite3.connect('fitara.db')
	if(conn):
		print("Opened database successfully")
	
	#preprocess = p.load(open('preprocess.pickle','rb'))

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
	print(str(y_test[0]))

	#getting name of docs
	docname1=request.form['docname1']
	docname2=request.form['docname2']
	docname3=request.form['docname3']
	modelname="RNN"

	#inserting into Table
	cur = conn.cursor()
	cur.execute("INSERT INTO results (Doc_name,Is_fitara,Model) VALUES (?,?,?)",(docname1,str(y_test[0]),modelname) )
	cur.execute("INSERT INTO results (Doc_name,Is_fitara,Model) VALUES (?,?,?)",(docname2,str(y_test[1]),modelname) )
	cur.execute("INSERT INTO results (Doc_name,Is_fitara,Model) VALUES (?,?,?)",(docname3,str(y_test[2]),modelname) )
	conn.commit()
	msg = "Record successfully added"
	conn.close()
	return render_template('status.html', msg = msg)

@app.route('/results',methods = ['GET','POST'])
def results():

	#getting Data out of the the table to display
	conn = sqlite3.connect('fitara.db')
	if(conn):
		print("Opened database successfully")
	cur = conn.cursor()
	cur.execute("select * from results")
	rows = cur.fetchall()
	conn.close()
	print(rows)
	return render_template("result.html",rows = rows)



if __name__ == '__main__':
    app.run(debug=True)