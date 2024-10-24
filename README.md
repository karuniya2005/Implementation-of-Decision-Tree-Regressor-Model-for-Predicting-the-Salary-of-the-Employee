# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. imprt required python libraries and load the CSV file using pandas
2. necessary preprocessing steps to be conducted
3. use labelencoder to encode the values
4. split the data set for training and test using train_test_split
5. import DecisionTreeRegressor to make decision
6. calculate the mean squared error and R2_score
7. do the prediction on the unsean values

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KARUNIYA M
RegisterNumber:  212223240068
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
print("First five rows:\n",data.head())
data.info()
print("Null Values:\n",data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print("Encoded Values:\n",data.head())

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
ypred=dt.predict(xtest)

from sklearn import metrics
mse=metrics.mean_squared_error(ytest,ypred)
print("Mean squared error: ",mse)

r2=metrics.r2_score(ytest,ypred)
print("R2_Score: ",r2)

print("Prediction: ",dt.predict([[5,6]]))
```

## Output:
![image](https://github.com/user-attachments/assets/a7a836e2-771e-4323-8342-6430a89ffc9f)

![image](https://github.com/user-attachments/assets/04d1a27b-bcbf-4526-99fb-b3abdfd295ed)

![image](https://github.com/user-attachments/assets/1e2632a2-f5b4-4e9b-a3d5-ffed590fffc6)

![image](https://github.com/user-attachments/assets/cca085e5-742e-4d88-8012-d03b947d29f6)

![image](https://github.com/user-attachments/assets/8c691d60-4ac1-4c73-a339-943020e42765)

![image](https://github.com/user-attachments/assets/64b0e61c-4426-4485-9d1b-38e3ce52d1b8)

![image](https://github.com/user-attachments/assets/0ff530ce-c754-4835-a527-26dc4a3c232d)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
