# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fcmeans import FCM
from seaborn import scatterplot as scatter
import skfuzzy as fuzz
from scipy import linalg
from sklearn import mixture
import itertools
import matplotlib as mpl


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

def gmm_cluster(Ehull, Form_eng):
    df = pd.DataFrame({
    'x':Ehull,
    'y':Form_eng})
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 11)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(df)
            bic.append(gmm.bic(df))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    plt.figure(figsize=(8, 8))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(df)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(df['x'][Y_ == i], df['y'][Y_ == i], .8, color=color)
    
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.xlabel('Stability (meV)')
    plt.ylabel('Formation Energy (meV)')
    plt.title('Selected GMM: full model, 9 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()
    
def c_mean_cluster(Ehull, Form_eng):
    #z = pd.concat([Ehull,Form_eng],join = 'outer',axis = 1)
    alldata = np.vstack((Ehull, Form_eng))

    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']


    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    fpcs = []
    

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later
        fpcs.append(fpc)

        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        for j in range(ncenters):
            ax.plot(Ehull[cluster_membership == j], Form_eng[cluster_membership == j], '.', color=colors[j])

        # Mark the center of each fuzzy cluster
        for pt in cntr:
            ax.plot(pt[0], pt[1], 'rs')

        ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        ax.axis('off')
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.r_[2:11], fpcs)
    ax2.set_xlabel("Number of centers")
    ax2.set_ylabel("Fuzzy partition coefficient")
    ax2.set_title('Fuzzy C-mean clustering algorithm')

    fig1.tight_layout()

gmm_cluster(z['Fuel yield μmol per g per cycle'],z['predicted Energy above hull'])
#c_mean_cluster(x,y)

