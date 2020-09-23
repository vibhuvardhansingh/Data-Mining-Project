# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



data = pd.read_excel('Fuel_yield.xlsx') 

data.plot(x='Fuel yield μmol per g per cycle', y='predicted Energy above hull', style='o')  

X = data['Fuel yield μmol per g per cycle'].values.reshape(-1,1)
Y = data['predicted Energy above hull'].values.reshape(-1,1)

x = data['Fuel yield μmol per g per cycle']
y = data['predicted Energy above hull']


df = pd.DataFrame({
    'x':x,
    'y':y})
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_


fig = plt.figure(figsize=(5, 3.5))

colmap = {1: 'r', 2: 'g', 3: 'b',4: 'y', 5: 'w'}

colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
    
plt.title('K-means clustering algorithm')
plt.xlabel('Fuel yield in ml per gm')
plt.ylabel('Stability')

plt.show()

'''
regressor = LinearRegression()
regressor.fit(X,Y)
'''



