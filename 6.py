import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.DataFrame({'Age':[22,25,29,32,35,40,45,50],
                 'Salary':[25000,30000,40000,42000,50000,55000,60000,62000],
                 'Experience_Years':[1,2,4,5,7,9,11,13],
                 'Department':['HR','Sales','Finance','Finance','Sales','HR','Sales','Finance']})
print("----- ORIGINAL DATASET -----");print(df)

df['Salary_per_Year']=df['Salary']/(df['Experience_Years']+1)
df['Age_Group']=pd.cut(df['Age'],[20,30,40,50],labels=['Young','Mid-age','Senior'])
print("\n----- AFTER CREATING NEW FEATURES -----");print(df)

df['Department']=LabelEncoder().fit_transform(df['Department'])
de=pd.get_dummies(df,columns=['Age_Group'],drop_first=True)
print("\n----- AFTER ENCODING CATEGORICAL VARIABLES -----");print(de)

X=de.drop('Salary',axis=1);y=df['Salary']
num=X.select_dtypes(int,float)
s=SelectKBest(score_func=f_regression, k=3)
s.fit(num,y)
sf=num.columns[s.get_support()]
print("\n----- SELECTED TOP FEATURES -----");print(sf)

Xa,Xb,ya,yb=train_test_split(num[sf],y,test_size=.3,random_state=42)
m=LinearRegression().fit(Xa,ya)
print(f"\nModel Accuracy using Selected Features: {m.score(Xb,yb):.2f}")

de.to_csv("Feature_Engineered_Data.csv",index=False)
print("\nFeature engineered dataset saved successfully!")
