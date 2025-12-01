import pandas as pd
import numpy as np

# Sample dataset
df=pd.DataFrame({
    'Name':['Amit','Bhavna','Chirag','Deepa','Esha','Farhan','Geeta','Harsh'],
    'Age':[22,25,np.nan,28,150,30,27,105],
    'Salary':[25000,30000,27000,np.nan,32000,-500,40000,35000]
})
print("Original Data:\n",df)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
print("\nAfter Handling Missing Values:\n",df)

df.loc[df['Age']>100,'Age']=df['Age'].median()
print("\nAfter Handling Outliers:\n",df)

df=df[df['Salary']>0]
print("\nAfter Data Validation:\n",df)

df.to_csv("cleaned_students.csv",index=False)
print("\nCleaned file saved as cleaned_students.csv")
