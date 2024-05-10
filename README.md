# Active Learning Project

This repository demonstrates the implementation of an active learning system using the Iris dataset. The project includes code for loading data, splitting it into training and pool sets, training a logistic regression model, and iteratively selecting the most uncertain samples to improve the model.

## Main Components of the Code

### Data Loading and Splitting

The data is loaded and split into two sets: an initial training set and a pool of unlabeled data.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data(test_size=0.95):
    data = load_iris()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=test_size, random_state=42)

`````

Function: load_data loads the Iris dataset using load_iris().
Splitting: It then splits this data into two parts using train_test_split():

A small initial training set (X_initial, y_initial).
A larger pool (X_pool, y_pool) from which the model will iteratively select new samples.
Initial Model Training

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_initial, y_initial)
`````
Model Initialization: A logistic regression model is created.
Training: The model is trained on the initial small dataset to start with some base knowledge.

```python
for i in range(iterations):
    probas = model.predict_proba(X_pool)
    uncertainties = np.max(probas, axis=1)
    query_idx = np.argmin(uncertainties)

    X_initial = np.vstack([X_initial, X_pool[query_idx]])
    y_initial = np.append(y_initial, y_pool[query_idx])

    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    model.fit(X_initial, y_initial)
    performance_history.append(accuracy_score(y_initial, model.predict(X_initial)))

`````

Uncertainty Measurement: The model calculates the probability of belonging to each class for all samples in the pool. The uncertainty of each sample is measured by the maximum probability across classes (i.e., how confident is the model about its prediction).
Querying Samples: The sample with the lowest maximum probability (most uncertain) is chosen for labeling (here, assumed to be labeled already for simplicity).
Data Update: This sample is added to the training set, and removed from the pool.
Model Retraining: The model is retrained on the updated training set.
Performance Tracking: After each iteration, the model's accuracy on the training set is calculated and recorded.
```python
print("Performance history:", performance_history)
`````
Output: The performance history is printed at the end of the process, showing how the model's accuracy changes with each iteration of adding new data.

## Summary

This process effectively demonstrates how active learning can leverage a model's uncertainty about the data to request labeling of the most informative samples first, thus potentially reducing the amount of data needed to achieve high performance. This iterative approach helps in efficiently utilizing data and computational resources, which is particularly useful in scenarios where labeling data is costly or time-consuming.


