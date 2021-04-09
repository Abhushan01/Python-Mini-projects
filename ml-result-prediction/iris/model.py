# importing libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#reading data from the csv file
lst=[]
df=pd.read_csv(r"C:\Users\Home\Desktop\Self-Learning\python_dev\django_dev\django_project02-IrisPredict\iris.csv")
print(df.head())

# spliting dataset into training and testing data
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.33,random_state=42)

# prediction using k-nearest neighbors method
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred1=knn.predict(X_test)
accuracy_knn=accuracy_score(y_test,pred1)*100
lst.append(accuracy_knn)
print(f"Accuracy score by knn method:{accuracy_knn}%")

#predictions using decision-tree method
classifier=DecisionTreeClassifier()
classifier.fit(X_train, y_train)
pred2=classifier.predict(X_test)
accuracy_dt=accuracy_score(y_test,pred2)*100
lst.append(accuracy_dt)
print(f"Accuracy score by decision tree method:{accuracy_dt}%")

#predictions using logistic regression method
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
pred3=logreg.predict(X_test)
accuracy_lr=accuracy_score(y_test,pred3)*100
lst.append(accuracy_lr)
print(f"Accuracy score by logistic regression method:{accuracy_lr}%")

#predictions using svm method
svc=svm.SVC()
svc.fit(X_train, y_train)
pred4=svc.predict(X_test)
accuracy_sv=accuracy_score(y_test,pred4)*100
lst.append(accuracy_sv)
print(f"Accuracy score by svm method:{accuracy_sv}%")

#predictions using randomForest method
model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)
pred5=model.predict(X_test)
accuracy_rf=accuracy_score(y_test,pred5)*100
lst.append(accuracy_rf)
print(f"Accuracy score by random Forest method:{accuracy_rf}%")

#taking user input to predict result from the ml models
val1=float(input("Enter val1:\t"))
val2=float(input("Enter val2:\t"))
val3=float(input("Enter val3:\t"))
val4=float(input("Enter val4:\t"))
result_pred=[[val1,val2,val3,val4]]

#checking for ml model having maximum value of accuracy score to make the prediction for the user
if accuracy_knn == max(lst):
    iris_pred=knn.predict(result_pred)
    print(iris_pred)

elif accuracy_dt ==max(lst):
    iris_pred=classifier.predict(result_pred)
    print(iris_pred)

elif accuracy_lr == max(lst):
    iris_pred=logreg.predict(result_pred)
    print(iris_pred)

elif accuracy_sv ==max(lst):
    iris_pred=svc.predict(result_pred)
    print(iris_pred)

else:
    iris_pred=model.predict(result_pred)
    print(iris_pred)