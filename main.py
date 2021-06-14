import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
import joblib
from guiCanvas import *
from knn import *

(train_x, train_labels), (test_x, test_labels) = mnist.load_data()
train_x = train_x / 255.
test_x = test_x / 255.

def GetMnist(dataSize = -1):
    global train_x, train_labels, test_x, test_labels

    if dataSize == -1:
        train_all = np.concatenate((train_x, test_x))
        labels_all = np.concatenate((train_labels, test_labels))
        return (train_all, labels_all)
    elif dataSize == 'train':
        return (train_x, train_labels)
    elif dataSize == 'test':
        return (test_x, test_labels)
    else:
        return (train_x[:dataSize], train_labels[:dataSize]), (test_x[:dataSize], test_labels[:dataSize])


def TestKnn(classifier, test_x, test_labels, newK = 0, show_time = True):
    if newK != 0:
        classifier.SetK(newK)
        
    if show_time:
        start_time = time.time()
        
    percentage = [True for (image, label) in zip(test_x, test_labels) if classifier.Predict(image) == label].count(True) / len(test_x) * 100

    if show_time:
        end_time = time.time()
        print("The process took: ", end_time - start_time, "seconds.")

    print("Success rate:", str(percentage) + '%', "    k = ", classifier.k)
    return percentage

def FindBestK(classifier, test_x, test_labels, rang = (1, 31), show_time = True):
    ks = list(range(rang[0], rang[1], 2))
    percs = []
    for k in ks:
        perc = TestKnn(classifier, test_x, test_labels, newK = k)
        percs.append(perc)
    
    plt.plot(ks, percs)
    plt.show()
    return ks[percs.index(max(percs))]



def TestNN(train_x, train_labels, test_x, test_labels, load = False):
    samples, nx, ny = train_x.shape
    train_x = train_x.reshape((samples, nx * ny))
        
    samples, nx, ny = test_x.shape
    test_x = test_x.reshape((samples, nx * ny))

    if not(load):
        NN = MLPClassifier(hidden_layer_sizes = (24, 24), activation = 'logistic')
        NN.fit(train_x, train_labels)
        joblib.dump(NN, 'NN.joblib')

    else:
        NN = joblib.load('NN.joblib')


    score = NN.score(test_x, test_labels)
    print("Success rate: ", score)

    predicted = NN.predict(test_x)
    print(classification_report(test_labels, predicted))
    plot_confusion_matrix(NN, test_x, test_labels, values_format="d", cmap="Blues")
    plt.show()
    

def Main():
    (data, labels) = GetMnist('train')
    (test_data, test_labels) = GetMnist('test')

    #Finding the best K for the KNN algorithm
    #cl = Knn(data, labels, Testing=True)
    #TestKnn(cl, test_data[:200], test_labels[:200])
    
    #FindBestK(cl, test_data[:100], test_labels[:100], rang=(1, 9))

    #Testing a Neural Network
    TestNN(data, labels, test_data, test_labels, load=True)

def UseGUI():
    gui = GUI()

if __name__ == "__main__":
    Main()