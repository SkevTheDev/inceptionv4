import tensorflow as tf
from CustomModelCheckPoint import CustomModelCheckpoint
from tensorflow.keras import datasets, layers, models
from keras import backend as K
from keras.optimizers import SGD

import matplotlib.pyplot as plt
class CNNModel(object):
    def create_model(self):
        # highest accuracy liveness on non-diffused iages- 16 (13,13) => 32 (7,7) => 64 (5,5) => 64 => 1 - acc = 94.2
        model = models.Sequential()
        model.add(layers.Conv2D(16, (15, 15), activation='relu', input_shape=(64, 64,1)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (7, 7), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dense(2))  # for sparse_categorial_crossentropy - then choose 2 neurons in next layer
        model.add(layers.Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(lr=0.0005, decay=1e-6)
        model.compile(optimizer= opt , loss= tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def load_model(self,model_file_name):
        model = tf.keras.models.load_model(model_file_name)
        K.set_value(model.optimizer.learning_rate, 0.1)
        return model

    def train_model(self, model, train_images,train_labels,test_images, test_labels, epochs, i, log, best):
        cbk = CustomModelCheckpoint(log, best)  # so that we can save the best model
        history = model.fit(train_images, train_labels, epochs=epochs, callbacks=[cbk], 
                        validation_data=(test_images, test_labels))
        #plt.plot(history.history['accuracy'], label='accuracy'+str(i))
        temp_best = max(history.history['val_accuracy'])
        if temp_best > best:
            best = temp_best
        #print("Last Best Val = " + str(best), file=log)
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy_'+str(i))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        #plt.show()
        plt.savefig('cnn_'+str(i)+'.png')
        return model, best

    def evaluate(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        return test_loss, test_acc