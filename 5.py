import pandas as pd,matplotlib.pyplot as plt

df=pd.DataFrame({'Employee_ID':[101,102,103,104,105,106,107,108],
                 'Department':['HR','Sales','Sales','Finance','HR','Finance','Sales','HR'],
                 'Experience_Years':[1,3,5,8,2,6,3,20],
                 'Salary':[25000,30000,40000,55000,28000,50000,32000,150000],
                 'Age':[22,25,28,32,23,30,26,55]})
print("----- ORIGINAL DATASET -----");print(df)

print("\n----- SUMMARY STATISTICS -----");print(df.describe())
print("\nEmployees per Department:\n",df['Department'].value_counts())
print("\nMissing Values per Column:\n",df.isnull().sum())

plt.figure(figsize=(6,4));df['Department'].value_counts().plot(kind='bar',color='skyblue',edgecolor='black');plt.title("Number of Employees per Department");plt.xlabel("Department");plt.ylabel("Employee Count");plt.grid(axis='y',linestyle='--',alpha=0.7);plt.show()
plt.figure(figsize=(6,4));plt.hist(df['Salary'],bins=5,color='orange',edgecolor='black');plt.title("Salary Distribution");plt.xlabel("Salary Range");plt.ylabel("Number of Employees");plt.grid(axis='y',linestyle='--',alpha=0.7);plt.show()
plt.figure(figsize=(6,4));plt.scatter(df['Age'],df['Salary'],color='green');plt.title("Age vs Salary");plt.xlabel("Age");plt.ylabel("Salary");plt.grid(True,linestyle='--',alpha=0.7);plt.show()

avg_salary=df.groupby('Department')['Salary'].mean()
print("\nAverage Salary per Department:\n",avg_salary)
threshold=df['Salary'].mean()+2*df['Salary'].std()
anomalies=df[df['Salary']>threshold]
print("\n----- POTENTIAL SALARY ANOMALIES -----\n",anomalies)

plt.figure(figsize=(6,4));plt.boxplot(df['Salary'],vert=False);plt.title("Salary Outlier Detection");plt.xlabel("Salary");plt.grid(axis='x',linestyle='--',alpha=0.7);plt.show()

df.to_csv("EDA_Processed_Data.csv",index=False)
print("\nEDA processing completed. File saved as 'EDA_Processed_Data.csv'")
