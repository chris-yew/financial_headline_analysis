import numpy as np
import pandas as pd
from flask import render_template, request, Flask
import model
from model.prediction import predict
from webscrap.scrap import get_headlines
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    result = None
    sentence = None
    headline_dict = None
    if request.method == 'POST':
        # print(request.form)
        try:
            sentence = request.form['statement']
            result = predict(sentence)
        except:
            sentence = None
            result = None

        try:
            request.form['generate']
            headline_dict = []
            headlines = get_headlines(10)
            for headline in headlines:
                res = dict(headline=headline, value=predict(headline))
                headline_dict.append(res)

        except:
            headline_dict = None

    return render_template('home_demo.html', sentence=sentence, result=result, headline_dict=headline_dict)


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
