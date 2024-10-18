# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: YOGESH. V
RegisterNumber:  212223230250
*/
```
```
import pandas as pd
data=pd.read_csv("Exp_8_Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![Screenshot 2024-10-18 185529](https://github.com/user-attachments/assets/a9cf359d-cfdd-481a-a406-08b6609a8b87)
### Info:
![Screenshot 2024-10-18 185548](https://github.com/user-attachments/assets/c4ba05bb-2ea9-4ca8-9d62-76677b5bdb21)
### Checking for null values:
![Screenshot 2024-10-18 185609](https://github.com/user-attachments/assets/397fa8a0-f58e-4f5d-94dc-90a38c61491b)

![Screenshot 2024-10-18 185620](https://github.com/user-attachments/assets/480edac8-7714-4d29-8d8e-641dc75a5cb0)
### Accuracy:
![Screenshot 2024-10-18 185632](https://github.com/user-attachments/assets/6d9126fb-c75a-4789-ae1e-211be48a8972)
### Predict:
![Screenshot 2024-10-18 185641](https://github.com/user-attachments/assets/028e0414-a786-4b36-97aa-b450fdaa0986)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
