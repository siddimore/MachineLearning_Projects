import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model

house_price = [245,312,279, 308, 199, 219,405, 324, 319, 255]
size = [1400, 1600,1700,1875,1100,1550,2350,2450,1425,1700]

size2= np.array(size).reshape((-1,1))

# fitting into model

regr = linear_model.LinearRegression()
regr.fit(size2,house_price)

print('Coefficients: \n', regr.coef_)
print('intercept: \n', regr.intercept_)


#############################
#formula obtained for the trained model
def graph(formula, x_range):
   x = np.array(x_range)
   y = eval(formula)
   plt.plot(x, y)
#plotting the prediction line
graph('regr.coef_*x + regr.intercept_', range(1000, 2700))
print regr.score(size2, house_price)
#############################

plt.scatter (size,house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()

print regr.predict([2000])