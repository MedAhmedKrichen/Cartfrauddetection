import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("creditcard.csv")

dataset.dropna(inplace=True)

from sklearn.model_selection import train_test_split



x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

plt.figure()
plt.hist(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model_svm=SVC()
model_svm.fit(x_train,y_train)
y_pred=model_svm.predict(x_test)

print(accuracy_score(y_test,y_pred))

from sklearn.neighbors import KNeighborsClassifier

model_knn=KNeighborsClassifier(n_neighbors=5)

model_knn.fit(x_train,y_train)
y_pred=model_knn.predict(x_test)
print(accuracy_score(y_test,y_pred))
