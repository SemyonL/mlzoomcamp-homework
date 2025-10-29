# Module 5. Deploying Machine Learning Models

## 5.1 Intro

Model is going to be deployed as web service.
- In Jupyter Notebook the model is trained and then saved to a file
- Web service is going to use saved model
- Separate service can ask for prediction from web service with the model

### Used stack:
- Flask is going to be use for creating web services.
- Pipenv is going to be used for environment.
- Docker is going to be used as container engine

## 5.2 Saving and loading model
Save the model:
```python
import pickle #lib for saving models

output_file = f'model_C={C}.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out) #Both dict vectorizer and model itseld is needed for using the model
```
Load the model:
```python
import pickle

with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
```
Script that trains the model:
- first: save jupyter notebook as python file
- remove and fix the script
- add some logging to the script if needed

## 5.3 Web services, Flask

```python
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```

## 5.4 Serving model with Flask

- Two web services
    - Prediction service (with the model)
    - Main service which uses prediction service

```python
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    result = { 'churn_probability': float(y_pred), 'churn': bool(churn) }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```

```python
import requests

url = ""
customer = {}
response = requests.post(url, json=customer)

response.json()
```

Production web server for flask:
```bash
pip install gunicorn

gunicorn --bind 0.0.0.0:9696 predict:app
```

## 5.5 Python virtual environment: Pipenv

RTM venv of python.

Other venv managers:
- conda
- pipenv (recommended)
- poetry

```bash
pipenv install XXX==ver

pipenv shell

pipenv run command to run
```

## 5.6 Docker

Container platform.

```bash
docker run -it --rm --entrypoint=bash python:3.8.12-slim
```

```DOCKERFILE
FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipefile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:9696", "predict:app"]
```
```bapredict:appsh
docker build -t zoomcamp-test .
``` zoomtest
```bash
docker run -it --rm -p 9696:9696 zoomtest
```

## 5.7 Deployment to AWS

AWS Elastic Beanstalk

## 5.8 Summary

- How to save the model to pickle
- How to load the model from pickle
- How to expose model with web service
- How to wrap web service into docker