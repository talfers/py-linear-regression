import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle


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

# Must train the model
# Vars MUST be in this order
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
# GET BEST MODEL VERSION
# start with best of 0
best = 0
for _ in range(50):
    # Vars MUST be in this order
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Create your linear regression algorithm
    linear = linear_model.LinearRegression()

    # Train your data
    linear.fit(x_train, y_train)

    # Create accuracy var
    acc = linear.score(x_test, y_test)
    print("Accuracy: \n", acc)

    if acc > best:
        best = acc
        # SAVING THE MODEL
        # Basically writing/saving pickle file
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

# Bring pickle file into script
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


# PREDICTION DATA
print("---------------------------------------\nDATA:\n---------------------------------------")

# Create predictions array (of array)
predictions = linear.predict(x_test)

# Loop over predictions and show
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Create graph with pyplot
plot = "G1"
style.use("ggplot")
pyplot.scatter(data[plot], data["G3"])
pyplot.xlabel(plot)
pyplot.ylabel("Final Grade")
pyplot.show()