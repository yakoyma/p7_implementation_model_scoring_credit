# Project 7: Implement a scoring model
import os
import numpy as np
import json
import pickle

from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True


def load_model():
    """
    This function loads a serialised machine learning file.
    """
    folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folder, 'model/model.pkl')
    with open(file_path, 'rb') as f:
       model = pickle.load(f)
       model = model['model']
    return model


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """This function is used for making prediction.
    and returns the score, the situation,
    and the status of the customer's application"""
    # Input data from the request of the dashboard
    request_json = request.get_json()
    print(request_json)
    data = []
    for key in request_json.keys():
        data.append(request_json[key])

    # Loading the model
    model = load_model()

    # Making prediction
    y_proba = model.predict_proba([data])[0][0]

    # Finding the situation of the customer (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.36
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Calculation of the customer's score
    score = int(y_class * 100)

    # Result of the credit application
    if customer_class == 1:
        situation = 'at risk'
        status = 'refused'
    else:
        situation = 'without risk'
        status = 'granted'

    # API response to the dashboard
    response = json.dumps(
        {'score': score, 'class': situation, 'application': status})
    return response, 200


if __name__ == '__main__':
    app.run(debug=True)



