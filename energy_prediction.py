import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from keras.models import Sequential                    #
from keras.layers import Dense, Dropout    #

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


def dnn(X_train, X_test, Y, train_data):
    x_scale, X_scale_test, y = data_conv(X_train, X_test, Y)
    model= Sequential()
    model.add(Dense(128, input_shape=(len(X_train.columns),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(np.array(x_scale[:train_data]), np.array(y[:train_data]), epochs=100, batch_size=10, verbose=1)
    loss,accuracy =model.evaluate(np.array(x_scale[:train_data]), np.array(y[:train_data]))
    print(loss, accuracy*100)
    result=model.predict_classes(np.array(X_scale_test[:train_data]))
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

def feature_vs_acc(X_train, X_test, y, y_test, no_of_units):
    X = []
    Y = []
    Z = []
    j = [10,20,30,40,50,60,70,80,90,100]
    for i in j:
        importance_x, importance_x_test = feature_selection(X_train, y,X_test, i)
        result, model_accuracy = dnn(importance_x, importance_x_test, y, no_of_units)
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
    
def ehull_pred(train, pred):
    df = pd.DataFrame({
    'x':train,
    'y':pred})

    plt.figure(figsize=(5, 5))
    plt.scatter(df['x'],df['y'])

    plt.show()


if __name__ == "__main__":

    X_scale, X_scale_test, y, ye, yf, y_test, ye_test, yf_test, Xc, Xd = wrap_data()
    importance_matrics, importance_matrics_test = feature_selection(X_scale, y,X_scale_test, 40)
    feature_vs_acc(X_scale, X_scale_test, y, y_test, 200)
    dnn_result, model_accuracy=dnn(importance_matrics, importance_matrics_test, y, 200)
    Ehull_vs_Foreng(ye, yf)
    confusion_matrix_result, accuracy, precision, recall, f1_score_result  = evaluation_metrics(dnn_result,y_test)
    





