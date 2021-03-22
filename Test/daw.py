import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pandas.read_csv("brystkræft_spørgeskema_undersøgelse.csv")

data = data[["Age", "Smokes", "Number of sexual partners", "Dx:HPV", "Hormonal Contraceptives",  "Dx:Cancer"]]

predict = "Dx:Cancer"

x = numpy.array(data.drop([predict], 1))
y = numpy.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""best = 0
for _ in range (30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open ("model.pickle", "wb",) as f:
            pickle.dump(linear, f) """

pickle_in = open ("Brystkræftmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Koefficenter: \n",  linear.coef_)
print("Intercept: \n",  linear.intercept_)

predictions = linear.predict(x_test)
print(predictions)

predictions = numpy.rint(predictions)

numpy.set_printoptions(suppress=True)
for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

Penis = "Number of sexual partners"
style.use("ggplot")
pyplot.scatter(data[Penis], data["Dx:Cancer"])
pyplot.xlabel(Penis)
pyplot.ylabel("Diagnose")
pyplot.show()