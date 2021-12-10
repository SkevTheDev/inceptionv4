import tensorflow as tf
import sys
from ReadData import ReadData
from CustomModelCheckPoint import CustomModelCheckpoint
from CNNModel import CNNModel
from tensorflow.keras import datasets, layers, models
from keras.optimizers import SGD
from InceptionV4Model import InceptionV4Model
import numpy as np
from keras import backend as K

import matplotlib.pyplot as plt


def main():
    #---set up path for training and test data (NUAA face liveness dataset)--------------
    readd = ReadData()
    clientdir = 'NormalizedFace_NUAA/ClientNormalized/'
    imposterdir = 'NormalizedFace_NUAA/ImposterNormalized/'
    client_train_normaized_file = 'NormalizedFace_NUAA/client_train_normalized.txt'
    imposter_train_normaized_file = 'NormalizedFace_NUAA/imposter_train_normalized.txt'
    
    client_test_normaized_file = 'NormalizedFace_NUAA/client_test_normalized.txt'
    imposter_test_normaized_file = 'NormalizedFace_NUAA/imposter_test_normalized.txt'

    #---------------read training, test data----------------
    train_images, train_labels = readd.read_data(clientdir, imposterdir, client_train_normaized_file, imposter_train_normaized_file)
    test_images, test_labels = readd.read_data(clientdir, imposterdir, client_test_normaized_file, imposter_test_normaized_file)

    log = open('log.txt', 'w')
    #print("hello", file=log)

    best = 0.9766

    for i in range(190,200):
        #--pick one of the following models for face liveness detection---
        #cnn = CNNModel()  # simple CNN model for face liveness detection---
        cnn = InceptionV4Model()  #Inception model for liveness detection
        if (i == 0):
            model = cnn.create_model()  # create and train a new model
            model, best = cnn.train_model(model, train_images,train_labels,test_images,test_labels, 30, i, log, best)
        else:
            model = cnn.load_model('model_CNN_v1.h5') #to use pretrained model  # or use model_CNN_v1.h5 for CNN model
            model, best = cnn.train_model(model,train_images,train_labels,test_images,test_labels, 30, i, log, best)  # to continue training from previously loaded model

            test_loss, test_acc = cnn.evaluate(model, test_images,  test_labels)
        print('iteration = ' + str(i) + ' ----- lr = ' + str(K.eval(model.optimizer.learning_rate)) +'', file=log)

    log.close()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
    
