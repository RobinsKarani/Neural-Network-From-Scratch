# Features, Examples, and `X.shape` in a Perceptron

## 1. What is a feature?

A **feature** is one measurable property or characteristic of an example.

Think of it as one **column** in a dataset. It is one piece of information the model uses to make a decision.

### Loan application example

If you are a loan officer, useful features might be:

| Feature               | Description                        |
| --------------------- | ---------------------------------- |
| Income                | How much money the applicant earns |
| Credit score          | A number showing creditworthiness  |
| Age                   | Applicant’s age                    |
| Loan amount requested | How much money they want to borrow |

Each feature can influence the final decision.

---

## 2. What is an example?

An **example** is one full row in the dataset. It is one item the model is trying to classify or predict.

Using the loan analogy, each loan application is one example.

### Example dataset

| Income | Credit Score | Age | Loan Amount | Approved? |
| ------ | ------------ | --- | ----------- | --------- |
| 50,000 | 650          | 30  | 10,000      | Yes       |
| 80,000 | 720          | 45  | 25,000      | Yes       |
| 20,000 | 500          | 22  | 5,000       | No        |

* Each **row** is one example.
* Each **column** is one feature, except the final label column.

---

## 3. Visualizing the dataset

```python
X = [
    [50000, 650, 30, 10000],   # Example 1
    [80000, 720, 45, 25000],   # Example 2
    [20000, 500, 22, 5000]     # Example 3
]
```

Here, the shape of `X` is `(3, 4)`:

* 3 examples
* 4 features per example

---

## 4. Why features matter

Features are what the model looks at before making a decision.

Good features help the model learn better patterns, and each feature gets a **weight** in the perceptron. That weight shows how important the feature is.

---

## 5. Understanding `n_samples, n_features = X.shape`

In a perceptron or any ML model, this line unpacks the shape of the dataset.

```python
n_samples, n_features = X.shape
```

### What `X` is

`X` is the input dataset, usually stored as a NumPy array.

Its structure is:

```text
(rows, columns) -> (samples, features)
```

### Example

```python
import numpy as np

X = np.array([
    [50000, 650],
    [80000, 720],
    [20000, 500]
])
```

| Income | Credit Score |
| ------ | ------------ |
| 50000  | 650          |
| 80000  | 720          |
| 20000  | 500          |

In this case:

* 3 rows -> 3 examples
* 2 columns -> 2 features

### What `.shape` returns

```python
X.shape
```

Output:

```python
(3, 2)
```

That means:

* 3 samples
* 2 features

### What the code does

```python
n_samples, n_features = X.shape
```

Python unpacks the tuple `(3, 2)` like this:

```python
n_samples = 3
n_features = 2
```

### Why this is needed

We use `n_features` to initialize the weights:

```python
self.weights = np.random.randn(n_features) * 0.01
```

If there are 2 features, we need 2 weights, one for income and one for credit score.

The perceptron then computes a score like this:

```python
score = (income * weight_income) + (credit_score * weight_credit) + bias
```

### Quick Python demo

```python
import numpy as np

X = np.array([
    [50000, 650],
    [80000, 720],
    [20000, 500]
])

print(X.shape)
```

Output:

```python
(3, 2)
```

### Why this is common in ML

Most ML code uses this pattern because the model needs to know:

* how many examples it has
* how many features each example contains

This helps initialize things like:

* weights
* matrices
* layers in neural networks
