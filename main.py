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

data = data[["Alder", "Køn",  "Bosted", "Fysisk_Aktivitet", "Vaegt", "Ryger", "Hormonel Praevention", "Indkomst", "Familiehistorie", "Kost", "Alkoholforbrug", "Brystkræft"]]

predict = "Brystkræft"
 
x = numpy.array(data.drop([predict], 1))
y = numpy.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)


print("Koefficenter: \n",  linear.coef_)
print("Intercept: \n",  linear.intercept_)

predictions = linear.predict(x_test)
print(predictions)

predictions = numpy.rint(predictions)

numpy.set_printoptions(suppress=True)
for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "Fysisk_Aktivitet"
style.use("ggplot")
pyplot.scatter(data[p], data["Brystkræft"])
pyplot.xlabel(p)
pyplot.ylabel("Diagnose")
pyplot.show()