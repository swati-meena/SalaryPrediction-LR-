import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r"/home/hp/Downloads/datasets/Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test   = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparison)

plt.scatter(x_test,y_test, color='red')  #real salary data
plt.plot(x_train, regressor.predict(x_train),color='blue') #regression line from training set
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print('Slope of the dataset :', m_slope)

c_intercept = regressor.intercept_
print('Constant of dataset :', c_intercept)

pred_12 = m_slope * 12 + c_intercept
print('Salary of 12 yrs Exp :', pred_12)

future_pred = m_slope * 20 + c_intercept
print('Salary of 20 yrs Exp :', future_pred)