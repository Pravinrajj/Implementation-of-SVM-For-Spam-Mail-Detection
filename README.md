# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Pravinrajj G.K
RegisterNumber:  212222240080
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### Result output
![242899205-a4611797-a5a5-4941-a4d6-67c9b0e1d128](https://github.com/Pravinrajj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117917674/363a750f-382f-4797-9d20-f3b930f785ab)

### data.head()
![242899300-a9be46a2-f290-4113-ab17-28c4cc7541fa](https://github.com/Pravinrajj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117917674/b1f89497-3d28-4050-ac33-4523bc596773)

### data.info()
![242899315-c5004333-df7f-4aec-ad5c-eccc67ee3463](https://github.com/Pravinrajj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117917674/0afaa521-e0f4-4ec6-8f8d-6b57402e68dd)

### data.isnull.sum()
![242899334-1de85604-7122-41a7-b0b1-998c1072fb5a](https://github.com/Pravinrajj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117917674/06ead582-b067-4bdb-b1c7-05441c1d9e23)

### Y_prediction()
![242899353-3ba82bfa-acf0-4213-ad35-9fa42cab6aea](https://github.com/Pravinrajj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117917674/ea12d3b7-44ca-4cc3-b372-bec51ce0cae5)

### Accuracy value
![242899416-afc3fd97-d327-4caf-ad5b-7c9c4f7f8c53](https://github.com/Pravinrajj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117917674/02e44f54-0419-4b41-89d2-737084ca7e99)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
