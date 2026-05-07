import numpy as np
from Logistic_Regression import LogisticRegression  # Assuming your class is defined in a file called Logistic_Regression.py
from sklearn.model_selection import train_test_split
from sklearn import datasets

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

model = LogisticRegression()
model.train(x_train, y_test)
y_pred = model.predict(x_test)  # Use predict method instead of predictions
acc = model.accuracy(y_pred, y_test)
print(acc)
