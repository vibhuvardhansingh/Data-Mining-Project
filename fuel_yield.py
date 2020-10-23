# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fcmeans import FCM
from seaborn import scatterplot as scatter
import skfuzzy as fuzz


from sklearn.mixture import GaussianMixture
model2=GaussianMixture(n_components=2,random_state=3425)



fcm = FCM(n_clusters = 5)


data = pd.read_excel('Fuel_yield.xlsx') 

data.plot(x='Fuel yield μmol per g per cycle', y='predicted Energy above hull', style='o')  

X = data['Fuel yield μmol per g per cycle'].values.reshape(-1,1)
Y = data['predicted Energy above hull'].values.reshape(-1,1)

x = data['Fuel yield μmol per g per cycle']
y = data['predicted Energy above hull']

z = pd.concat([x,y],join = 'outer',axis = 1)
model2.fit(z)
uu= model2.predict(z)

fcm.fit(z)

df = pd.DataFrame({
    'x':x,
    'y':y})
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fcm_labels  = fcm.u.argmax(axis=1)

f, axes = plt.subplots(1, 2, figsize=(11,5))
scatter(x, y, ax=axes[0],hue = labels)
plt.title('fuzzy-c-means algorithm')
scatter(x, y, ax=axes[1], hue=fcm_labels)
#scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
plt.show()

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']


#cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(z, 5, 2, error=0.005, maxiter=1000, init=None)

fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        z, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(x[cluster_membership]==j,
                y[cluster_membership]==j, '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')
    
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")

fig1.tight_layout()


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(labels,fcm_labels)

print(cm) 

fig = plt.figure(figsize=(5, 3.5))

colmap = {1: 'r', 2: 'g', 3: 'b',4: 'y', 5: 'w'}

colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
    
plt.title('K-means clustering algorithm')
plt.xlabel('Fuel yield in micro-mole per gm per cycle')
plt.ylabel('Stability (meV)')

plt.show()

'''
regressor = LinearRegression()
regressor.fit(X,Y)
'''



