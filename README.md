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
import pandas as pd
from sklearn.tree import  plot_tree
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```
## Output:
### Head:
![Screenshot 2024-10-18 035531](https://github.com/user-attachments/assets/25553681-1b66-4db7-bce4-13bd9b4aa7ad)
### Mean Squared Error:
![Screenshot 2024-10-18 035539](https://github.com/user-attachments/assets/a5724c2f-a6df-49c7-89e3-bc5c7af2a226)
### Predicted Value:
![Screenshot 2024-10-18 035552](https://github.com/user-attachments/assets/c1e2dd68-04db-4e1a-89da-af2eabb19d39)
### Decision Tree:
![Screenshot 2024-10-18 035606](https://github.com/user-attachments/assets/e79f688d-903d-4abc-8f46-da3cbfea074b)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
