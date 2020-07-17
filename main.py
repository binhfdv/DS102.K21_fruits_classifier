import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functions import load_Data, rbgToGrayAndResize
from models import model_KNN, model_SVM, joblib
from sklearn.metrics import accuracy_score



if __name__ == "__main__":

    cwd = os.getcwd()
    output_path = cwd+'\outputs\models'
    try:
        os.makedirs(output_path)
    except:
        pass
    

    X_train, Y_train, X_test, Y_test = [], [], [], []


    X_train, Y_train = load_Data(X_train, Y_train, 'Training_new')
    X_test, Y_test = load_Data(X_test, Y_test, 'Test_new')

    print(len(X_train))
    print(len(Y_train))
    print(len(X_test))
    print(len(Y_test))
    
    X_train = rbgToGrayAndResize(X_train)
    X_test  = rbgToGrayAndResize(X_test)

    nsample, nx, ny = X_train.shape
    X_train = X_train.reshape((nsample, nx*ny))

    nsample2, nx, ny = X_test.shape
    X_test = X_test.reshape((nsample2, nx*ny))

    SC = StandardScaler()
    X_train = SC.fit_transform(X_train)
    X_test = SC.transform(X_test)


    print(X_test.shape)

    model_KNN(X_train, Y_train, output_path)

    f = open('accuracy.txt', 'w')

    knn_minkowski_3 = joblib.load(output_path+'\knn_minkowski_3.pkl')
    accuracy1 = accuracy_score(Y_test, knn_minkowski_3.predict(X_test))
    f.write('knn_minkowski_3: '+str(accuracy1)+'\n')
    print('acc1---done')
    knn_minkowski_5 = joblib.load(output_path+'\knn_minkowski_5.pkl')
    accuracy2 = accuracy_score(Y_test, knn_minkowski_5.predict(X_test))
    f.write('knn_minkowski_5: '+str(accuracy2)+'\n')
    print('acc2---done')
    knn_minkowski_7 = joblib.load(output_path+'\knn_minkowski_7.pkl')
    accuracy3 = accuracy_score(Y_test, knn_minkowski_7.predict(X_test))
    f.write('knn_minkowski_7: '+str(accuracy3)+'\n')
    print('acc3---done')
    svm = joblib.load(output_path+'\svm.pkl')
    accuracy4 = accuracy_score(Y_test, svm.predict(X_test))
    f.write('svm: '+str(accuracy4)+'\n')
    print('acc4---done')

    knn_euclidean_3 = joblib.load(output_path+'\knn_euclidean_3.pkl')
    accuracy1 = accuracy_score(Y_test, knn_euclidean_3.predict(X_test))
    f.write('knn_euclidean_3: '+str(accuracy1)+'\n')
    print('acc1---done')
    knn_euclidean_5 = joblib.load(output_path+'\knn_euclidean_5.pkl')
    accuracy2 = accuracy_score(Y_test, knn_euclidean_5.predict(X_test))
    f.write('knn_euclidean_5: '+str(accuracy2)+'\n')
    print('acc2---done')
    knn_euclidean_7 = joblib.load(output_path+'\knn_euclidean_7.pkl')
    accuracy3 = accuracy_score(Y_test, knn_euclidean_7.predict(X_test))
    f.write('knn_euclidean_7: '+str(accuracy3)+'\n')
    print('acc3---done')

    f.close()

