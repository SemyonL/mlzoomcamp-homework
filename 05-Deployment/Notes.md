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

## Workshop

### Pipelines:

```python
from skilearn import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)

train_dict = df[categorical + numerical].to_dict(orient='records')
y_train = df.churn

pipeline.fit(train_dict, y_train)

with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

churn = pipeline.predict_probna(customer)[0,1]

```

### Notebook conversion:

```bash
jupyter nbconvert --to-script nobebook.ipynb
```

### FastAPI
```bash
pip install fastup uvicorn
```
```python
from fastapi import FastAPI
import uvicorn
import pickle
from typing import Dict, Any

app = FastAPI(title="churn prediction")

@app.get("/ping")
def ping():
    return 'PONG'

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0,1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    churn = predict_single(customer)
    return {
        "churn_probablity": churn,
        "churn": bool(churn >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
```

`localhost:9696/docs` - FastAPI API docs for the current app

### uvicorn with hot reload
```bash
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

### Calling the service
Requests library is used. See [previous notes](#54-serving-model-with-flask).

### Validating requests with FastAPI

```python
from typing import Literal
from pydantic import BaseModel, Field

#request
class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

#response
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )
```

### Virtual environments with uv

uv - virtual environment manager written in Rust (extremely fast).
```bash
pip install uv

uv init # empty project initialization
# After project init is over, toml file can be edited with metadata

uv add scikit-learn fastapi uvicorn # add dependencies

uv add --dev requests # development only dependencies

uv run command # run command in virtual environment
```

### Using docker

Refer to the [module notes](#56-docker).

```Dockerfile
FROM python:3.12.1-slim-bookworm

#RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="app/.venv/bin:$PATH"

COPY ".python-version", "pyproject.toml", "uv.lock" "./"

RUN uv sync --lock # sync dependencies and install all the libs

COPY "predict.py", "model.bin", "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
# There is no need to use uvrun because it is only one python environment in docker
```
```bash
docker build -t predict-churn .
docker run -it --rm -p 9696:9696 predict-churn
```

### Deployment to Fly.io

```bash
curl -L https://fly.io/install.sh | sh

fly auth signup
fly launch --generate-name
fly deploy

# Remove deployment
fly apps destroy <app-name>
```