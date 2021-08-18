import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, render_template, request, url_for
from sklearn.feature_extraction.text import CountVectorizer

## TEXT PROCESSING
df = pd.read_csv('data_cnn_cleaned.csv')

# load model
model = pickle.load(open('model-smote.pkl','rb'))

# vectorize the 'text' data
vectorizer = CountVectorizer(min_df=2, ngram_range=(1,4))
fit_vec = vectorizer.fit(df['cleaned'])

# class
sentiment_dict = {0:'neutral',1:'positive',2:'negative'}


## ROUTE TO HTML
# define template (html file) location
app = Flask(__name__, template_folder=os.getcwd(), static_url_path='/static')

# route to home page
@app.route('/')
def main():
    return render_template('home2.html')

@app.route('/prediction')
def predict():
    return render_template('home.html')

@app.route('/maps')
def map():
    return render_template('maps.html')

@app.route('/output')
def outputs():
    return render_template('output.html')

# route to result page
@app.route('/result',methods=['POST'])
def result():
    # get input text
    input_text = np.array([request.form['a']])
    # encode text
    encode_text = fit_vec.transform(input_text)
    # prediction
    prediction = model.predict(encode_text)

    return render_template('home.html', data=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)