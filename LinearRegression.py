import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set=pd.read_csv("Salary_Data.csv")
x=data_set.iloc[:,:-1]
y=data_set.iloc[:,1]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=0)
reg=LinearRegression()
reg.fit(x_train, y_train)

#y_pred_x=reg.predict(x_train)
y_pred=reg.predict(x_test)


mtp.scatter(x_test, y_test, color="blue")
mtp.scatter(x_test, y_pred, color="red")
mtp.title("Salary vs Experience")
mtp.xlabel("Years of Experience")
mtp.ylabel("Salary")
mtp.show()