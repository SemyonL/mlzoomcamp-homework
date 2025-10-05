# Module 1. Introduction to Machine Learning.

## 1.1 Introduction to Machine Learning

Introduces a car price prediction web-site. How to determine the right price for the car which is not too low and not too high, so the car is sold with the maximum price and minimal time.

How machine learning can be applied to it based on initial data about cars, i.e., production year, marque, mileage, etc.

The pipeline: Data -> Expert / ML -> Patterns

Initial data is called **features**. Prediction - **target**.

Combine features and target to learn patterns from the data *to train* a **model**.

Models do not predict the exact value, but the average is going to be correct.

Machine Learning is the process of extracting patterns from data.

Features + Model -> Predictions. E.g. car data + Model = Car price prediction.

## 1.2 ML vs Rule Based System

Spam detection system example. Rule based system to detect spam, e.g., if sender contains specific email, or subject have specific text, or body contains specific words etc.

Machine learning approach:
1. Get the data / users may provide it with spam button.
2. Define features. Rules can be used as first featurs.
3. Take initial data and fill out features and targets data.
4. Create a model.
5. Use predictions result and make a threshold for final decision.

## 1.3 Supervised Machine Learning

Essence behind superviced machine learning: the algorithm learns by example. There is source data and there is target data used for pattern prediction learning.

Essential things from recap of e-mail spam filtering:
1. Feature extraction into feature matrix (X)
2. Target varialbe into vector (y)

#### Formal definition of supervised machine learning:
g(X) ~= y
* g(X) - model
* X - features matrix
* y - target vector

### Types of supervicesd ML:
* Regression (car/house price prediction)
* Classification (outputs a category, e.g. car from image, spam)
    * Multiclass (classify images to a set of categories)
    * Binary classification (into 2 categories)
* Ranking (scoring, e.g. recommendations in e-commerce, search engines)

## 1.4 CRISP-DM
Cross-Industry Standard Process for Data Mining

ML project workflow:
1. Understand the problem.
2. Collect the data.
3. Train the model.
4. Use it.

Main ML project steps:
1. Bussiness Understanding
    * Understand the problem and measures of success. Is ML really needed?
2. Data understanding
    * Analyse data sources available and collect the data.
    * Is data good enough? Is it reliable? Is it large enough? Is more data needed?
3. Data preparation
    * Transform data into form usable by ML algorithms
    * Data clean up, pipelines, convert data into tabular form, extract features
4. Modelling
    * Training the model: try different model and select the suitable one.
    * Add more features if necessary, fix data issues when found.
5. Evaluation
    * Measure how good is the model.
    * Is the goal reached? Is the right thing solved/measured?
    * When needed go back and adjust.
6. Deployment (nowdays comes with Evaluation step combined)
    * Online evaluations with live users.
    * Ensure the quality and maintablity with monitoring.

and then iterate.

## 1.5 Modelling step: Model Selection

Try different models and choose the best one: model selection process.

1. Choose 80% of entire data set for training. Remaining 20% is used for validation.
2. Extract feature matrix X and y from the training data and train the model g.
3. Extract feature matrix X_v and y_v from validation data set.
4. Then take different model types g_1, g_2, g_3 and calculate probablity based on y_v ~= g(X_v) for each g in g_1, g_2, g_3...

Multiple comparisons problem: perform the same comparison many times and evaluate it on the same data set.

#### Validation & Test

60% - Train, 20% - Validation, 20% - Test

When the model selected on validation set must be tested with test data set to see if validation was correct.

1. Split the data set.
2. Train the model.
3. Validate the model with validation data set.
4. Select the best model.
5. USing the best model, apply it on test data set.
6. Check that the model is good.
7. (additional) retrain the model with traning data set + validation data set. Then test it with test data set.

## 1.6 Environment setup (GitHub code space / local environment)

### GitHub codespace setup:
1. Create new codespace in GitHub.
2. Install python modules: 
```bash
pip install numpy pandas scikit-learn seaborn jupyter
```
3. Run jupyter notebook
```bash
jupyter notebook
```
4. Create nootbook and use it. It is possible to use it inside VS Code too.

### Local machine setup (Windows ARM)
To avoid network lag working with codespaces it is possible to run notebooks locally.
However, jupyter is not shipped with Windows ARM binaries (yet?) and it requires build from source which fails due to nuget cli usage inside rust source. The solution I've found is to use WSL to setup virtual environment in Linux ARM envuronment.

0. Login into WSL. (I use Ubuntu with python3 and python3-venv installed)
1. Create virtual environment.
```bash
python3 -m venv .venv
```
2. Activate it.
```bash
. .venv/bin/activate
```
3. Check the active virtual environment.
```bash
which python
```
4. Install python modules
```bash
pip install numpy pandas scikit-learn seaborn jupyter
```
5. Connect VS Code to WSL. Open notebook and select the virtual environment.

P.S. It takes around 750 MiB of space.

## 1.7 Intro to NumPy

```python
import numpy as np

#Create arrays
np.zeros(5)
np.ones(10)
np.full(10, 2.5)
np.array([1,2,3,4,5])
np.arrange(3, 10) #[3,4...,9,10]
np.linspace(0, 1, 11) #array of size 11 filled with numbers from 0 to 1

#Multi-dementional arrays
np.zeroes((5, 2))
np.array([[1,2,3], [4,5,6]])
n[1] #access one whole row
n[:, 1] #access whole column

np.random.rand(5, 2) #2-dementional array filled with pseudo-random distribution in [0; 1]
np.random.seed(10) #set the seed of generator
np.random.randn(5,2) #normal distribution
np.random.randint(low=0, high=100, size=(5,2))

#Element operations
a = np.arrange(5)
a + 1 #Adds 1 to each element in array, same for other math operations
b = (10 + (a * 2)) ** 2 #Complex math is also possible
a + b #Combine arrays and other operations
a >= 2 #Return array with per-element comparison or other arrays
a[a > b] #a > b - returns array of booleans, the outer a[] return all elements that are True

#Summarizing operations
a.min()
a.sum()
a.mean()
a.std() #standard deviation

a.min(axis=0) #min for each column, axis=1 for each row etc.

a.shape[0] #Number of columns, 1 - rows, n - other demensions
```

## 1.8 Linear Algebra Refresher

What's refreshed:
* Vector operations
* Dot product or vector-vector multiplication (a number) in numpy: u.dot(v)
* Transpose
* Matrix-vector multiplication in numpy: U.dot(v), where U is 2-demensional
* Matrix-matrix multiplication in numpy: U.dot(V), U,V - matrices
* np.eye(x) gives identity matrix
* Matrix inverse: np.linalg.inv(V), V - square matrix

## 1.9 Intro to Pandas

DataFrame - main data structure: a table.
```python
#DataFrame creation
pd.DataFrame(data, columns=columns)
pd.DataFrame(data) #data is a list of dictionaries, or just list of lists then columns are numbered
df.heads(n=X) #Several first n rows/records
df.SeriesName #or df['SeriesName'] return specific column
df[['Series1', 'Series2', 'Series3']] #return specified series
df['new_col'] = [1,2,3,4,5] #add new column
del df['col'] #remove column
df.index #range of element indecies
df.loc[1] #get element with index 1
df.loc[[1,3,5]] #multiple elements
#index can be changed to letters
df.iloc[1] #positional index, ignores frame's index
df.reset_index(drop=False) #resets index to sequence of numbers and return new data frame
df[df['Year'] >= 2015] #as in numpy it filters using the array of bools

#String operators
df['Series'].str.lower() #Return data frame with lower case values

#Summarizing operators
df.Series.describe() #returns useful summary stats
df.Series.nunique()

df.isnull() #True if the value is missing
df.isnull().sum() #See missing values in all columns

#Grouping
df.groupby('Series').Series2.mean() #Group by Series, and return average for Series2 in each group
df.Serives.values #original velues array

#Conversion
df.to_dict(orient='records') #return a list of dictionaries
```

## Summary
* Intro to ML, features, target variable, Model
* Rules based vs ML systems
* Supervised ML, g(X) ~= y, g - model, X - features, y - target
* ML pipeline or workflow
* Modeling and Model Selection
* Environment setup
* NumPy
* Linear Algebra refresh
* Pandas