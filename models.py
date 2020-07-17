import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def model_KNN(X_train, Y_train, path):
    # minkowski distance
    knn_minkowski_3 = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski')
    knn_minkowski_3.fit(X_train, Y_train)
    joblib.dump(knn_minkowski_3, path+'\knn_minkowski_3.pkl')
    print('1 done')

    knn_minkowski_5 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski')
    knn_minkowski_5.fit(X_train, Y_train)
    joblib.dump(knn_minkowski_5, path+'\knn_minkowski_5.pkl')
    print('2 done')

    knn_minkowski_7 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski')
    knn_minkowski_7.fit(X_train, Y_train)
    joblib.dump(knn_minkowski_7, path+'\knn_minkowski_7.pkl')
    print('3 done')

    #euclidean distance
    knn_minkowski_3 = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean')
    knn_minkowski_3.fit(X_train, Y_train)
    joblib.dump(knn_minkowski_3, path+'\knn_euclidean_3.pkl')
    print('4 done')

    knn_minkowski_5 = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
    knn_minkowski_5.fit(X_train, Y_train)
    joblib.dump(knn_minkowski_5, path+'\knn_euclidean_5.pkl')
    print('5 done')

    knn_minkowski_7 = KNeighborsClassifier(n_neighbors = 7, metric = 'euclidean')
    knn_minkowski_7.fit(X_train, Y_train)
    joblib.dump(knn_minkowski_7, path+'\knn_euclidean_7.pkl')
    print('6 done')

def model_SVM(X_train, Y_train, path):
    svm = SVC(kernel='linear')
    svm.fit(X_train, Y_train)
    joblib.dump(svm, path+'\svm.pkl')

def model_LGR(X_train, Y_train, path):
    lgr = LogisticRegression(random_state = 0, max_iter=70000)
    lgr.fit(X_train, Y_train)
    joblib.dump(lgr, path+'\lgr.pkl')
