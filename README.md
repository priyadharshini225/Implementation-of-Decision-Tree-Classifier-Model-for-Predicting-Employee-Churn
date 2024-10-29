# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load and inspect the dataset (Employee.csv) to understand its structure and check for missing values.
   
2. Encode categorical variables, such as "salary," to prepare the data for modeling.
   
3. Split the data into training and testing sets with an 80-20 ratio.

4. Train a DecisionTreeClassifier using the "entropy" criterion on the training data.

5. Evaluate model accuracy and make predictions on test data and a sample input.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PRIYADHARSHINI S
RegisterNumber: 212223240129

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"]) 
data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred) 
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]]) 
*/
```

## Output:
![Screenshot 2024-10-12 131340](https://github.com/user-attachments/assets/c1e25e42-7124-48e8-ae17-8b5d79531778)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
