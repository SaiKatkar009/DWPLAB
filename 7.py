import pandas as pd,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.DataFrame({'City':['Mumbai','Delhi','Pune','Delhi','Chennai','Mumbai','Pune','Chennai'],'Experience_Years':[1,3,5,2,4,6,7,8],'Department':['HR','Finance','HR','Sales','Finance','HR','Sales','Finance'],'Selected':[0,1,1,0,1,1,0,1]})
print("----- ORIGINAL DATASET -----");print(df)

X=df[['City','Experience_Years','Department']];y=df['Selected']
le=LabelEncoder();Xl=X.copy();Xl['City']=le.fit_transform(Xl['City']);Xl['Department']=le.fit_transform(Xl['Department'])
print("\n----- AFTER LABEL ENCODING -----");print(Xl)

ct=ColumnTransformer(transformers=[('e',OneHotEncoder(drop='first'),['City','Department'])],remainder='passthrough')
Xo=ct.fit_transform(X);Xo=pd.DataFrame(Xo.toarray() if hasattr(Xo,"toarray") else Xo)
print("\n----- AFTER ONE-HOT ENCODING -----");print(Xo)

Xa,Xb,ya,yb=train_test_split(Xl,y,test_size=.3,random_state=42)
Xc,Xd,yc,yd=train_test_split(Xo,y,test_size=.3,random_state=42)

m1=LogisticRegression().fit(Xa,ya);m2=DecisionTreeClassifier(random_state=42).fit(Xa,ya)
p1=m1.predict(Xb);p2=m2.predict(Xb)
mo1=LogisticRegression().fit(Xc,yc);mo2=DecisionTreeClassifier(random_state=42).fit(Xc,yc)
po1=mo1.predict(Xd);po2=mo2.predict(Xd)

print("\n----- MODEL ACCURACY COMPARISON -----")
print(round(accuracy_score(yb,p1),2),round(accuracy_score(yb,p2),2),round(accuracy_score(yd,po1),2),round(accuracy_score(yd,po2),2))

pd.concat([Xl,y],axis=1).to_csv("Label_Encoded_Data.csv",index=False)
pd.concat([Xo,y.reset_index(drop=True)],axis=1).to_csv("OneHot_Encoded_Data.csv",index=False)
print("\nEncoded datasets saved successfully!")
