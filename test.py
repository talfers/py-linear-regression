import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# Read Data
data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())

# ATTRIBUTES
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

# LABEL
predict = "G3"

# Create array of Attributes/Features
x = np.array(data.drop([predict], 1))

# Create array of Labels
y = np.array(data[predict])

# Vars MUST be in this order
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Create your linear regression algorithm
linear = linear_model.LinearRegression()

# Train your data
linear.fit(x_train, y_train)

# STATS
print("---------------------------------------\nSTATS:\n---------------------------------------")

# Create accuracy var
acc = linear.score(x_test, y_test)
print("Accuracy: \n", acc)

# Print m
print("Coefficients: \n", linear.coef_)

# Print b
print("Intercept: \n", linear.intercept_)

# PREDICTION DATA
print("---------------------------------------\nDATA:\n---------------------------------------")

# Create predictions array (of array)
predictions = linear.predict(x_test)

# Loop over predictions and show
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])