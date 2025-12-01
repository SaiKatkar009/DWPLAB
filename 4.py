import pandas as pd

dfc=pd.DataFrame({'Customer_ID':[101,102,103,104,105],
                  'Customer_Name':['Amit','Bhavna','Chirag','Deepa','Esha'],
                  'City':['Mumbai','Delhi','Pune','Mumbai','Chennai']})
dfs=pd.DataFrame({'Customer_ID':[101,102,103,103,106],
                  'Purchase_Amount':[2500,1800,2200,2200,1500],
                  'Purchase_Date':['2025-10-01','2025-10-03','2025-10-05','2025-10-05','2025-10-07']})

print("----- Dataset 1: Customers -----\n",dfc)
print("\n----- Dataset 2: Sales -----\n",dfs)

m=pd.merge(dfc,dfs,on='Customer_ID',how='inner')
print("\n----- Merged Data (Inner Join) -----\n",m)

d=m.drop_duplicates()
print("\n----- After Deduplication -----\n",d)

agg=d.groupby('City')['Purchase_Amount'].sum().reset_index()
print("\n----- Aggregated Data (Total Purchase per City) -----\n",agg)

agg.to_csv("Final_Integrated_Data.csv",index=False)
print("\nFinal integrated data saved as 'Final_Integrated_Data.csv'")
