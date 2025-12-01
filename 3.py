import pandas as pd, numpy as np

df=pd.DataFrame({'Name':['  Amit ','Bhavna','Chirag@123','Deepa','Esha_'],
                 'Email':['amit@gmail.com','bhavna@yahoo.com','chirag@gmail.com','deepa@hotmail.com','esha@outlook.com'],
                 'Joining_Date':['2025-01-05','2025-02-10','2025-03-15','2025-02-25','2025-04-20'],
                 'Department':['Sales','HR','Sales','Finance','HR'],
                 'Salary':[25000,30000,27000,35000,32000]})
print("----- ORIGINAL DATA -----\n",df)

df['Name']=df['Name'].str.strip().str.upper().str.replace(r'[^A-Z]','',regex=True)
df['Email_Domain']=df['Email'].str.split('@').str[1]
print("\n----- AFTER STRING MANIPULATION -----\n",df)

gmail_users=df[df['Email'].str.contains(r'@gmail\.com$')]
print("\n----- GMAIL USERS (FILTERED USING REGEX) -----\n",gmail_users)

df['Joining_Date']=pd.to_datetime(df['Joining_Date'])
df['Join_Year']=df['Joining_Date'].dt.year
df['Join_Month']=df['Joining_Date'].dt.month
df['Join_Day']=df['Joining_Date'].dt.day
today=pd.Timestamp('2025-11-03')
df['Experience_Days']=(today-df['Joining_Date']).dt.days
print("\n----- AFTER DATE/TIME HANDLING -----\n",df)

pivot_df=df.pivot_table(values='Salary',index='Department',aggfunc=np.mean).reset_index()
print("\n----- PIVOT TABLE: AVERAGE SALARY PER DEPARTMENT -----\n",pivot_df)

melted_df=pd.melt(pivot_df,id_vars=['Department'],value_vars=['Salary'],var_name='Metric',value_name='Value')
print("\n----- MELTED DATA (RESHAPED FORMAT) -----\n",melted_df)

df.to_csv("Transformed_Data.csv",index=False)
print("\nTransformed data saved as 'Transformed_Data.csv'")
