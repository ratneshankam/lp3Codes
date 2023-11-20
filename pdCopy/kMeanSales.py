import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Ratnesh/OneDrive/Desktop/LP3 Codes/sales_data_sample.csv', encoding='ISO-8859-1')

df.head()

df.dtypes

df.isna().sum()

df.info()

df_drop  = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS','POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1)

df.head()

df.shape

df.isna().sum()

df.dtypes

country = pd.get_dummies(df['COUNTRY'])
productline = pd.get_dummies(df['PRODUCTLINE'])
Dealsize = pd.get_dummies(df['DEALSIZE'])

df = pd.concat([df,country,productline,Dealsize], axis = 1)

df.head()

df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE']
df = df.drop(df_drop, axis=1)

df.dtypes

df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes

df.dtypes

df.drop('ORDERDATE', axis=1, inplace=True)

df.dtypes

from sklearn.cluster import KMeans

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init=10)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(20,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kmeanModel = KMeans(n_clusters=3)
y_kmeans = np.array(kmeanModel.fit_predict)