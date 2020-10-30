import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from fcmeans import FCM
from seaborn import scatterplot as scatter
from sklearn import mixture
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn.model_selection import train_test_split


from keras.models import Sequential              #
from keras.layers import Dense, Dropout, LSTM    #

def read_data(cont_name, disc_name):
    cdata = pd.read_csv(cont_name, index_col=0)
    ddata = pd.read_csv(disc_name, index_col=0)
    Xc = cdata[cdata.columns[0:-2]]
    Xd = ddata[ddata.columns[0:-2]]
    y = cdata[cdata.columns[-2]]
    yl = y.copy()
    y2 = cdata[cdata.columns[-1]]
    # label the instance
    # Energy above 40 meV is considered to be unstable
    y[y <= 40] = 1
    y[y > 40] = 0
    return Xc, Xd, y, yl, y2

def data_process(Xc, Xd, Xc_model, Xd_model):
    p = 0.00
    sel = VarianceThreshold()
    sel.fit(Xc_model)
    Xc1 = Xc.loc[:, sel.variances_ > p * (1 - p)]
    sel.fit(Xd_model)
    Xd1 = Xd.loc[:, sel.variances_ > p * (1 - p)]
    # print('Removed {} feature from {} continuous features'.format(Xc.shape[1] - Xc1.shape[1], Xc.shape[1]))
    # print('Removed {} feature from {} discrete features'.format(Xd.shape[1] - Xd1.shape[1], Xd.shape[1]))
    X_total = pd.concat([Xc1, Xd1], axis=1)
    return X_total

def data_scale(X_total, X_total_model):
    scaler = preprocessing.StandardScaler().fit(X_total_model)
    feature_names = list(X_total)
    X_scaled = scaler.transform(X_total)
    return pd.DataFrame(X_scaled,columns=feature_names)

def data_conv(X_scale, X_scale_test, y):
    x_train=X_scale.values.tolist()             #
    x_test=X_scale_test.values.tolist()         #
    y0=y.values.tolist()                        #
    y1= [int(i) for i in y0]
    return x_train, x_test, y1


def dnn(X_train, X_test, Y):
    x_scale, X_scale_test, y = data_conv(X_train, X_test, Y)
    model= Sequential()
    model.add(Dense(128, input_shape=(len(X_train.columns),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(np.array(x_scale), np.array(y), epochs=100, batch_size=10, verbose=1)
    loss,accuracy =model.evaluate(np.array(x_scale), np.array(y))
    print(loss, accuracy*100)
    result=model.predict_classes(np.array(X_scale_test))
    #result= pd.Dataframe(dnn_predict, columns=['dnn prediction'])
    return result, accuracy

def cut_highEs(X_features, yl, ye):
    # remove outliers
    X_s = X_features.loc[ye < 400]
    yl_s = yl[ye < 400]
    return X_s, yl_s

def wrap_data():

    ctrain = 'c_0_train.csv'
    ctest = 'c_0_test.csv'
    dtrain = 'd_0_train.csv'
    dtest = 'd_0_test.csv'

    Xc, Xd, y, ye, yf = read_data(ctrain, dtrain)
    X_total = data_process(Xc, Xd, Xc, Xd)
    X_scale = data_scale(X_total, X_total)

    Xc_test, Xd_test, y_test, ye_test, yf_test = read_data(ctest, dtest)
    X_total_test = data_process(Xc_test, Xd_test, Xc, Xd)
    X_scale_test = data_scale(X_total_test, X_total)

    ye = ye.reset_index()['EnergyAboveHull']
    y = y.reset_index()['EnergyAboveHull']
    yf = yf.reset_index()['Formation_energy']

    return X_scale, X_scale_test, y, ye, yf, y_test, ye_test, yf_test, Xc, Xd

def evaluation_metrics(y_pred, y_test):
    
    confusion_matrix_result = confusion_matrix(y_pred,y_test)
    accuracy = accuracy_score(y_pred,y_test)
    precision = precision_score(y_pred,y_test)
    recall = recall_score(y_pred,y_test)
    f1_score_result = f1_score(y_pred,y_test)
    
    return confusion_matrix_result, accuracy, precision, recall, f1_score_result

def feature_selection(X_train, Y_train,X_test, no_of_features):
    model = ExtraTreesClassifier()
    model.fit(X_train, Y_train)
    importance = model.feature_importances_
    importance_matrics = pd.DataFrame({'features':X_train.columns, 'importance':importance})
    importance_matrics = importance_matrics.sort_values(by='importance', ascending = False)
    features = np.array((importance_matrics[:no_of_features]['features'].index).tolist())
    return X_train.iloc[:,features], X_test.iloc[:,features]

def feature_vs_acc_dnn(X_train, X_test, y, y_test):
    X = []
    Y = []
    Z = []
    j = [10,20,30,40,50,60,70,80,90,100]
    for i in j:
        importance_x, importance_x_test = feature_selection(X_train, y,X_test, i)
        result, model_accuracy = dnn(importance_x, importance_x_test, y)
        confusion_matrix_result, accuracy, precision, recall, f1_score_result  = evaluation_metrics(result,y_test)
        X.append(i)
        Y.append(model_accuracy)
        Z.append(accuracy)
        
    df = pd.DataFrame({
        'x':X,
        'y':Y,
        'z':Z})
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(df['x'], df['y'], df['z']);
    ax.set_xlabel('No. of features')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Test Acc. vs Model Acc. vs No. of features')


    #plt.plot(df['x'],df['y'])
    #plt.plot(df['x'],df['z'])

    return 

def feature_vs_acc_rnn(X_train, X_test, y, y_test):
    X = []
    Y = []
    Z = []
    j = [10,20,30,40,50,60,70,80,90,100]
    for i in j:
        importance_x, importance_x_test = feature_selection(X_train, y,X_test, i)
        result, model_accuracy = rnn_lstm(importance_x, importance_x_test, y)
        confusion_matrix_result, accuracy, precision, recall, f1_score_result  = evaluation_metrics(result,y_test)
        X.append(i)
        Y.append(model_accuracy)
        Z.append(accuracy)
        
    df = pd.DataFrame({
        'x':X,
        'y':Y,
        'z':Z})
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(df['x'], df['y'], df['z']);
    ax.set_xlabel('No. of features')
    ax.set_ylabel('Model Accuracy')
    ax.set_title('Test Acc. vs Model Acc. vs No. of features')

    #plt.plot(df['x'],df['y'])
    #plt.plot(df['x'],df['z'])

    return 
    
def Ehull_vs_Foreng(Ehull, Form_eng):
    df = pd.DataFrame({
    'x':Ehull,
    'y':Form_eng})
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_


    plt.figure(figsize=(15, 5))

    colmap = {1: 'r', 2: 'g', 3: 'b',4: 'y', 5: 'w'}

    colors = list(map(lambda x: colmap[x+1], labels))

    plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
    for idx, centroid in enumerate(centroids):
        plt.scatter(*centroid, color=colmap[idx+1])
    
    plt.title('K-means clustering algorithm')
    plt.xlabel('Ehull')
    plt.ylabel('Formation Energy')

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
    
def c_mean_cluster_graph(Ehull, Form_eng):
    df = pd.DataFrame({
    'x':Ehull,
    'y':Form_eng})
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    z = pd.concat([Ehull,Form_eng],join = 'outer',axis = 1)
    fcm = FCM(n_clusters = 6)
    fcm.fit(z)
    fcm_labels  = fcm.u.argmax(axis=1)
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    scatter(Ehull, Form_eng, ax=axes[0], hue = labels)
    plt.title('fuzzy-c-means algorithm')
    scatter(Ehull, Form_eng, ax=axes[1], hue=fcm_labels)
    plt.show()
    
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
    plt.figure(figsize=(8, 6))
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
    plt.title('Selected GMM: full model, 7 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()
    
def ehull_pred(train, pred):
    df = pd.DataFrame({
    'x':train,
    'y':pred})

    plt.figure(figsize=(11, 8))
    plt.scatter(df['x'],df['y'])
    plt.xlabel('Stability(meV)')
    plt.ylabel('Formation Energy(meV)')
    plt.title('Stabilit vs Formation Energy for Perovskites')

    plt.show()
    
def merging_test_train(X_scale, X_scale_test, y, y_test):
    merged_dataset_x = pd.concat([X_scale,X_scale_test]).reset_index(drop=True)
    merged_dataset_y = pd.concat([y,y_test]).reset_index(drop=True)
    merged = pd.concat([merged_dataset_x,merged_dataset_y], axis = 1)
    return merged_dataset_x, merged_dataset_y, merged
    
def rnn_lstm(X_train, X_test, Y):
    x_scale, X_scale_test, y = data_conv(X_train, X_test, Y)
    model= Sequential()
    model.add(LSTM(40, input_shape=(len(X_train.columns),1),activation='relu',return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    a=np.array(x_scale)
    b=a[:,:,np.newaxis]
    #print(b.shape)
    
    m = model.fit(b, np.array(y), epochs=15, batch_size=1000, verbose=1)
    loss,accuracy =model.evaluate(b, np.array(y))
    print(loss, accuracy*100)
    a_test=np.array(X_scale_test)
    b_test=a_test[:,:,np.newaxis]
    b_test = model.predict_classes(b_test)
    return b_test ,accuracy

def split_test_train(merged_x,merged_y,test_size):
    X_scale, X_scale_test, y, y_test = train_test_split(merged_x,merged_y, test_size=test_size, random_state=45)
    return X_scale, X_scale_test, y, y_test

if __name__ == "__main__":

    X_scale, X_scale_test, y, ye, yf, y_test, ye_test, yf_test, Xc, Xd = wrap_data()
    merged_x, merged_y, merged = merging_test_train(X_scale, X_scale_test, y, y_test)
    no_of_features = 40
    test_size = 0.35 # test size in percent
    X_scale, X_scale_test, y, y_test = split_test_train(merged_x, merged_y, test_size)
    importance_matrics, importance_matrics_test = feature_selection(X_scale, y,X_scale_test, no_of_features)
    #feature_vs_acc_dnn(X_scale, X_scale_test, y, y_test)
    #feature_vs_acc_rnn(X_scale, X_scale_test, y, y_test)
    #dnn_result, dnn_model_accuracy=dnn(importance_matrics, importance_matrics_test, y)
    #rnn_result, accuracy = rnn_lstm(importance_matrics,importance_matrics_test,y)
    #Ehull_vs_Foreng(ye, yf)
    #c_mean_cluster(ye, yf)
    #c_mean_cluster_graph(ye,yf)
    #gmm_cluster(ye,yf)
    #ehull_pred(ye, yf)
    #confusion_matrix_result_dnn, accuracy_dnn, precision_dnn, recall_dnn, f1_score_result_dnn  = evaluation_metrics(dnn_result,y_test)
    #confusion_matrix_result_rnn, accuracy_rnn, precision_rnn, recall_rnn, f1_score_result_rnn  = evaluation_metrics(rnn_result,y_test)






