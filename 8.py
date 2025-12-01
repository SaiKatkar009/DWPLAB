import pandas as pd,matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE

X,y=make_classification(n_samples=1000,n_features=5,n_informative=3,n_redundant=0,n_clusters_per_class=1,weights=[.9,.1],random_state=42)
df=pd.DataFrame(X,columns=[f'F{i}'for i in range(1,6)]);df['T']=y
print("----- ORIGINAL CLASS DISTRIBUTION -----");print(df['T'].value_counts())
plt.figure();df['T'].value_counts().plot(kind='bar');plt.title('Original Class Distribution');plt.show()

Xt,Xe,yt,ye=train_test_split(X,y,test_size=.3,random_state=42)
m1=LogisticRegression().fit(Xt,yt);p1=m1.predict(Xe)
print("\n----- MODEL PERFORMANCE (Before SMOTE) -----");print(confusion_matrix(ye,p1));print(classification_report(ye,p1))

Xr,yr=SMOTE(random_state=42).fit_resample(Xt,yt)
print("\n----- CLASS DISTRIBUTION AFTER SMOTE -----");print(pd.Series(yr).value_counts())
plt.figure();pd.Series(yr).value_counts().plot(kind='bar');plt.title('Balanced Class Distribution');plt.show()

m2=LogisticRegression().fit(Xr,yr);p2=m2.predict(Xe)
print("\n----- MODEL PERFORMANCE (After SMOTE) -----");print(confusion_matrix(ye,p2));print(classification_report(ye,p2))
