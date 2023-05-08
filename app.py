import numpy as np
import pandas as pd
from flask import render_template, request, Flask
import model
from model.prediction import predict
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    # return "<p>Hello, World!</p>"
    if request.method == 'POST':
        sentence = request.form['statement']
        print('SENTENCE:', sentence)
        result = predict(sentence)
        return render_template('home_demo.html', sentence=sentence, result=result)
    return render_template('home_demo.html')


'''
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        sentence = request.form['statement']
        print('SENTENCE:', sentence)
        # return '<p> success <p>'
        return render_template('home_demo.html', sentence=sentence)

#action="{{url_for ('prediction')}}"
'''

if __name__ == '__main__':
    app.run()
