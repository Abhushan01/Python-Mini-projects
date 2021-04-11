# importing libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#reading data from csv file
lst=[]
df=pd.read_csv(r"C:\Users\Home\Desktop\Self-Learning\python_dev\django_dev\django_project03-CoviChecker\Corona-report_cln.csv")


#creating designing matrix X and target vector y
X=df.drop("corona_result", axis=1)
y=df['corona_result']
print(X.head(),"\n")
print(y.head())
#splitting dataset into training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.33,random_state=42)


# predictions using decision-tree method
classifier=DecisionTreeClassifier()
classifier.fit(X_train, y_train)
pred1=classifier.predict(X_test)
accuracy_dt=accuracy_score(y_test,pred1)*100
print(f"Accuracy score by decision tree method is {accuracy_dt}%")
lst.append(accuracy_dt)

# predictions using randomForest method
model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)
pred2=model.predict(X_test)
accuracy_rf=accuracy_score(y_test,pred2)*100
print(f"Accuracy score by Random Forest method is {accuracy_rf}%")
lst.append(accuracy_rf)

# predictions using logistic regression 
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
pred3=logreg.predict(X_test)
accuracy_lr=accuracy_score(y_test,pred3)*100
print(f"Accuracy score by logistic regression method is {accuracy_lr}%")
lst.append(accuracy_lr)


#cough
val1=input("Cough: (yes/no)").lower()
#fever
val2=input("Fever: (yes/no)").lower()
#sore-throat
val3=input("Throat: (yes/no)").lower()
#shortness-of-braethe
val4=input("Breathe: (yes/no)").lower()
#headache
val5=input("Headache: (yes/no)").lower()
#age-60-and-above (0,1)
val6=input("Age : (yes/no)").lower()
#gender (0,1)
val7=input("Gender: (male/female)").lower()
#test-indication (0,1,2)
val8=input("test-indication: (other, contact with confirmed, abroad)").lower().replace(" ", "")

if 'yes' in (val1, val2, val3, val4, val5, val6):
    val1=val2=val3=val4=val5=val6=1
elif 'no' in (val1,val2,val3,val4,val5,val6):
     val1=val2=val3=val4=val5=val6=0

if val7=='male':
    val7=1
elif val7=='female':
    val7=0

if val8=='other':
    val8=0
elif val8=='contactwithconfirmed':
    val8=1
elif val8=='abroad':
    val8=2

form_values=[[val1,val2,val3,val4,val5,val6,val7,val8]]

#checking for ml model having maximum value of accuracy score to make the prediction for the user
def covid_result_pred(form_input):
    if accuracy_dt ==max(lst):
        result_pred=classifier.predict(form_input)
     

    elif accuracy_rf ==max(lst):
        result_pred=model.predict(form_input)
      

    else:
        result_pred=logreg.predict(form_input)
        
    
    return print(result_pred)

covid_result_pred(form_values)
     
