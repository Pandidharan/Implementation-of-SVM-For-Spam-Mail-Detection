# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import necessary libraries (chardet, pandas, sklearn modules).

2. Detect Encoding: Use chardet to find the encoding of spam.csv.

3. Load Dataset: Read spam.csv with the detected encoding using pandas.

4. Explore Data: Check the first few rows and identify any null values.

5. Define Features and Labels: Set v2 (messages) as x and v1 (spam/ham) as y.

6. Split Data: Use train_test_split to divide data into training and testing sets.

7. Vectorize Text: Convert text data to numerical format using CountVectorizer.

8. Initialize SVM: Instantiate the SVC model.

9. Train Model: Fit the SVM model on the training data.

10. Make Predictions and Evaluate: Predict on test data, calculate accuracy, and print results.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Pandidharan.G.R
RegisterNumber: 212222040111

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv('spam.csv',encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train=cv.fit_transform(X_train)
X_test=cv.transform(X_test)
from sklearn.svm import SVC
svc =SVC()
svc.fit(X_train,Y_train)
y_pred = svc.predict(X_test)
print("Predictions:\n",y_pred)
from sklearn import metrics
accuracy = metrics.accuracy_score(Y_test,y_pred)
print("accuracy:",accuracy)
```

## Output:
### Predictions:
![image](https://github.com/user-attachments/assets/39cc0a56-7262-4f66-b0ac-f515f7265824)
### Accuracy:
![image](https://github.com/user-attachments/assets/a6481484-4e8c-48a8-9ddf-75a6f27667b1)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
