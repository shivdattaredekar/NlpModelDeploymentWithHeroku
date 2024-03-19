from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb')) # This is done to make sure that whatever input we get from the user we 
#are able to convert it into vectors and pass it to my predict function as input to give a proper output  
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray() # Pass the msg to transform hence it will converted to vector
		my_prediction = clf.predict(vect) 
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True,port=5001)