import numpy as np
import pandas as pd
import sys
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from keras.models import Sequential                    #
from keras.layers import Dense, Activation, Dropout    #

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

def select_features(index_file, select_n, X_total):
    indices_data = pd.read_csv(index_file, names=['order'])
    ''' b = indices_data.sort_values(by = ['order'])'''
    indices = np.array(indices_data['order'].tolist())
    ''' v = indices.copy()
    v = v.sort()'''
    selected = indices[:select_n]
    X_features = X_total.iloc[:, selected]
    return X_features

def data_conv(X_scale, X_scale_test, y):
    x_train=X_scale.values.tolist()             #
    x_test=X_scale_test.values.tolist()         #
    y0=y.values.tolist()                        #
    y1= [int(i) for i in y0]
    return x_train, x_test, y1

def classification(X_scale, X_scale_test, y):
    clf = ExtraTreesClassifier(criterion='entropy', bootstrap=False, max_leaf_nodes=None,
                               max_features=43, class_weight='balanced', # min_impurity_split=0.1 is removed
                               min_samples_split=5, min_samples_leaf=1, max_depth=18, n_estimators=115)
    X_features1 = select_features('RFE_clf_indices.txt', 70, X_scale)
    X_features_test1 = select_features('RFE_clf_indices.txt', 70, X_scale_test)

    clf.fit(X_features1, y)
    stability_predict = clf.predict(X_features_test1)
    clf_result = pd.DataFrame(stability_predict, columns=['predicted stability'])
    return clf_result

def dnn(X_train, X_test, Y):
    x_scale, X_scale_test, y = data_conv(X_train, X_test, Y)
    model= Sequential()
    model.add(Dense(128, input_shape=(791,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(np.array(x_scale[:200]), np.array(y[:200]), epochs=100, batch_size=10, verbose=1)
    loss,accuracy =model.evaluate(np.array(x_scale[:200]), np.array(y[:200]))
    print(loss, accuracy*100)
    result=model.predict_classes(np.array(X_scale_test[:200]))
    #result= pd.Dataframe(dnn_predict, columns=['dnn prediction'])
    return result

def cut_highEs(X_features, yl, ye):
    # remove outliers
    X_s = X_features.loc[ye < 400]
    yl_s = yl[ye < 400]
    return X_s, yl_s

def reg_EaH(X_scale, X_scale_test, ye):
    reg = KernelRidge(kernel='rbf', alpha=0.007, gamma=0.007)
    X_features2 = select_features('RFE_eah_indices.txt', 70, X_scale)
    X_features_test2 = select_features('RFE_eah_indices.txt', 70, X_scale_test)

    X_s, ye_s = cut_highEs(X_features2, ye, ye)
    reg.fit(X_s, ye_s)
    y_predict = reg.predict(X_features_test2)
    EaH_predict = pd.DataFrame(y_predict, columns=['predicted Energy above hull'])
    return EaH_predict


def reg_FE(X_scale, X_scale_test, yf, ye):
    reg = KernelRidge(kernel='rbf', alpha=0.00464, gamma=0.0215)
    X_features3 = select_features('stability_fe_indices.txt', 20, X_scale)
    X_features_test3 = select_features('stability_fe_indices.txt', 20, X_scale_test)
    X_s, yf_s = cut_highEs(X_features3, yf, ye)
    reg.fit(X_s, yf_s)
    y_predict = reg.predict(X_features_test3)
    FE_predict = pd.DataFrame(y_predict, columns=['predicted Formation Energy'])
    return FE_predict

def write_result(testfile, output, clf_result, EaH_predict, FE_predict):
    test_data = pd.read_excel(testfile)
    raw_composition = test_data[['Material Composition', 'A site #1', 'A site #2',
                                 'A site #3', 'B site #1', 'B site #2', 'B site #3',
                                 'X site', 'Number of elements']]
    result = pd.concat([raw_composition, clf_result, EaH_predict, FE_predict], axis=1)
    result.to_excel(output, index=None)

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

def feature_selection(X_train, Y_train, no_of_features):
    model = ExtraTreesClassifier()
    model.fit(X_train, Y_train)
    importance = model.feature_importances_
    importance_matrics = pd.DataFrame({'features':X_train.columns, 'importance':importance})
    importance_matrics = importance_matrics.sort_values(by='importance', ascending = False)
    features = np.array((importance_matrics[:no_of_features]['features'].index).tolist())
    return X_train.iloc[:,features]
    
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

    trainfile = 'perovskite_DFT_EaH_FormE.xlsx' if len(sys.argv)<=1 else sys.argv[1]
    testfile = 'newCompound.xlsx' if len(sys.argv)<=2 else sys.argv[2]
    id = 0 if len(sys.argv)<=3 else sys.argv[3]

    X_scale, X_scale_test, y, ye, yf, y_test, ye_test, yf_test, Xc, Xd = wrap_data()
    importance_matrics = feature_selection(X_scale, y, 25)
    clf_result = classification(X_scale, X_scale_test, y)                        
    dnn_result=dnn(X_scale, X_scale_test, y)      #
    EaH_predict = reg_EaH(X_scale, X_scale_test, ye)
    FE_predict = reg_FE(X_scale, X_scale_test, yf, ye)
    Ehull_vs_Foreng(ye, yf)
    ehull_pred(ye_test.to_numpy(), EaH_predict['predicted Energy above hull'].to_numpy())
    confusion_matrix_result, accuracy, precision, recall, f1_score_result  = evaluation_metrics(dnn_result,y_test)
    output = 'energy_prediction_result.xlsx'
    write_result(testfile, output, clf_result, EaH_predict, FE_predict)
    





