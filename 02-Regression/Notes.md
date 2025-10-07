# Module 2. Machine Learning for Regression

This is Module 2 notes. It includes basic keypoints based on my knowledge. The note include just main point and pieces of code for remembering purpose. The code is 

## 2.1 Car price prediction project

Project plan:
* Prepare data
* Train linear regression
* Implement linear regression
* Evaluate the model's quality
* Feature engineering
* Regularization
* Use of the model

## 2.2 Data preparation

```python
pd.read_csv('data.csv')

# Step 1. Make data consistent
df.columns
df.columns = df.columns.str.replace(' ', '_').str.lower() # DO NOT FORGET! that data are not changed in-place, reassignment is needed.
df.dtypes # contains column types
columns = list(df.dtypes[df.dtypes == 'object'].index) #index contains column names
df[col] = df[col].str.lower().str.replace(' ', '_') #replaes column data with updated values
#all object values (strings) should be also normalized into consistent state
```

## 2.3 Exploratory data analysis

```python
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique()
    print()

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

%mathplotlib_inline

sns.histplot(df.msrp, bins=50) #Histogram
sns.histplot(df.msrp[df.msrp < 100000], bins=50)
# tail distribution confuses the model
# Convert long tail distribution to logarithmic

np.log1p([0,1,10,1000,100000]) #adds 1 to every number and takes log function, useful because log 0 doesn't exist

price_logs = np.log1p(df.msrp)
sns.historyplot(price_logs, bins=50)

# Normal distribution is better for ML models!

# Looking for missing values
df.isnull().sum()
```

## 2.4 Setting up the validation framework

Recap: split the data set in 3: for training, for validation, for test (60%. 20%. 20%)

```python
n_val = int(len(df) * 0.2)
n_test = n_val
n_tran = len(df) - n_val - n_test

df.iloc[:10] #first 10 indexed records
```
Data sets needs to be suffeled to remove any accidential order

```python
idx = np.arrange(n)
np.random.suffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train.reset_index(drop=True) #do for all 3

y_train = np.log1p(df_train.msrp.values) #do for all 3

del df_train['msrp'] #delete non-logarithmic values to avoid using it by accident
```

## 2.5 Linear regression (LR)

LR - model for solving regression problem to predict numbers

$g(X) \approx y$

$g(x_i) \approx y_i$ - single observation, one car ($x_i$) and its price ($y_i$)

$x_i = (x_{i1}, x_{i2}, ..., x_{in})$

$g(x_{i1}, ..., x_{in}) \approx y_i$

### Example:
$x_i = [453, 11, 86]$ , horse power, fuel efficency, popularity

$g(x_i) \approx y_i$

LR formula: $g(x_i) = w_0 + w_1*x_{i1} + w_2*x_{i2} + w_3 * x_{i3}$ , where $w_i$ - weights, $w_0$ - bias term (prediction when no information is available)

$g(x_i) = w_0 + \displaystyle\sum_{j=1}^{3} w_j*x_{ij}$

$w_0 = 7.17, w = [0.01, 0.04, 0.002]$

$g(x_i) = 12.312$

$exp(g) = 222347.222...$

```python
#userful functions
np.log1p(X) #log (X + 1)
np.expm1(X) #exponent (X - 1)
```

## 2.6 Linear regression vector form

$ g(x_i) = w_0 + \displaystyle\sum_{j=1}^{n}x_{ij} \cdot w_j = w_0 + x_i^Tw = $

$ = w_0 \cdot x_{i0} + x_i^Tw$ - vector form, where always $x_{i0} = 1$, $x_{i0}$ is added with value 1 to have two vectors.

$ w = [w_0, w_1, ..., w_n], x_i = [x_{i0}, x_{i1}, ..., x_{in}] $

$w^Tx_i = x_i^Tw = w_0 $

$X$ - features matrix (n+m)
$$
\begin{bmatrix} 1 & x_{11} & ... & x_{1n} \\ 1 & x_{21} & ... & x_{2n} \\ 1 & ... & ... & ... \\ 1 & x_{m1} & ... & x_{mn} \end{bmatrix}
\cdot
\begin{bmatrix} w_0 \\ w_1 \\ ... \\ w_n \end{bmatrix}
=
\begin{bmatrix} x_1^T \cdot w \\ x_2^T \cdot w \\ ... \\ x_m^T \cdot w \end{bmatrix}
$$
Features $X$ * weights $w$ = Prediction $y_p$ : $ X \cdot w = y_p $

## 2.7 Training a linear regression model

$ g(X) = X \cdot w \approx y $

Assume $X^{-1} \cdot X \cdot w = X^{-1} \cdot y  \implies  w = X^{-1} \cdot y$

$X^{-1}$ doesn't always exists. so it is possibe to use Gram matrix:

$X^T X w = X^T y$ , where $X^T X$ - Gram matrix, which is square matrix with $(n+1) * (n+1)$ dimensions, and inverse matrix for this exists, therefore:

$(x^T X)^{-1} X w = (X^T X)^{-1} X^T y \implies I w = (X^T X)^{-1} X^T y \implies w = (X^T X)^{-1} X^T y$

```python
X = np.array(X)
ones = np.ones(X.shape[0]) #unit vector [1,1,...,1] for biase term
X = np.column_stack([ones, X])
XTX = X.T.dot(X) #(X^T X)
XTX_inv = np.linalg.inv(XTX) #(X^T X)^{-1}
w_full = XTX_inv.dot(X.T).dot(y) #(X^T X)^{-1} X^T y
```

## 2.8 Car price baseline model

1. Use all numeric columns from the data set
```python
df_train.dtypes
base = ['col1', 'col2', 'coln']
#DO NOT FORGET to check and fill missing values.
X_train = df_train[base].values.fillna(0) #replacing with zeroes, not the best scenario but works fine. Means features are ignored.
w0, w = train_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)
sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_train, color='blue', alpha=0.5, bins=50)
```

## 2.9 Root Mean Squared Error (RMSE)

RMSE $ = \sqrt{\displaystyle\frac1m \displaystyle\sum_{i=1}^m ( g(x_i) - y_i )^2 }$ , where $g(x_i)$ - prediction, $y_i$ - actual value

$ ( g(x_i) - y_i )^2 $ - squared difference between prediction and actual value or *squared error*

$ \displaystyle\frac1m ( g(x_i) - y_i )^2$ - *mean squared error*

## 2.10 Validating the model

Use RMSE on validation data set and trained data set to see the error.

## 2.11 Simple feature engineering

Instead of year, age feature is created
```python
df.year.max() #find maximum year
df.copy() #copy data set when needed
df['age'] = 2017 - df.year #calculate the age and add it to column
```

## 2.12 Categorical variables

Any non-numerical values. Usual ways to encode it: divide one categorical columnd into several binary columns. I.e., per each value new binary column is created. The value 1 assigned to each record which contanins the column category, 0 is assigned to every other record.

```python
for v in [2, 3, 4]:
    df_train['num_doors_%s' % v] = (df_train.number_of_doors == v).astype('int') #example for one column
```

## 2.13 Regularization

$ w = (X^T X)^{-1} \cdot X^T \cdot y $

In some cases $(X^T X)^{-1}$ does not exists. If features have the same values or duplicated columns then inverse does **not** exists. The reason usually that data is not clean or incorrect.
To solve that issue and ensure that $(X^T X)^{-1}$ exists, it is possible to add small number to the main diagonal of $X$, e.g. $+0.0001$

```python
XTX = XTX + 0.01 * np.eye(XTX.shape[0]) #this is regularisation or controling the weights they they won't go into cosmos. r = 0.01
np.linalg.inv(XTX) #(X^T X)^{-1}
```

## 2.14 Tuning the model

Using validation data set to find the best r (regularisation value). Make a list [10, 1, 0.1, 0.01, 0.001, 0.0001] and select best RMSE value and check bias $w_0$.

## 2.15 Using the model

### Training the final model.

Data sets:  
- Train
- Validation
- Test

Steps:
- Train model on Train data set
- Apply validation to get RMSE
- Tune the model
- Train the model on Train + Validation data sets, *this is the final model*
- Test it on Test data set.

```python
df_full_train = pd.concat([df_train, df_val])
df_full_train.reset_index(drop=True)
X_full_train = prepare_X(df_full_train)

y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_regularised(X_full_train, y_full_train, r=0.001)
```

### Using the model
to predict the car price.

```python
df_small = pd.DataFrame([car]) #Get single car in data frame
X_small = prepare_X(df_small) #Get features from 1 car in df
y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0] #It is just one car, don't need list
np.expm1(y_pred) #Price prediction
```

## Summary

- Price car prediction project
- Data cleanup: make data uniform
- Data analysis
- Remove long tail distribution by applying logarithmic
- Fix missing data
- Set the validation framework
- Linear regression for single example
- Linear regression in vector form
- Training linear regression model
- Baseline model training with basic feaetures
- RMSE
- Validation framework based on RMSE
- Simple feature engineering (age from year)
- Categorical valirables converted to binary columns
- Numerical instabilities and solving it with regularisation
- Tuning the model to find best regularisation parameter
- Using the model