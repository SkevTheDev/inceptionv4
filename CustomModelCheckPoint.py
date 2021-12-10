import tensorflow as tf

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, log, best):
        # set the lastvalacc to a higher value (best that was found) in the 
        # previous training so that it looks for a better model than the last one
        self.lastvalacc = best  #0.9796 #orig inception v4 0.9788, # 0.9703 Inception v4 diffused # last best
        self.log = log
        print("\n\nLast Best Val = " + str(self.lastvalacc), file=self.log)
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        
        #print(f"epoch: {epoch}, train_acc: {logs['accuracy']}, valid_acc: {logs['val_accuracy']}")

        if logs['val_accuracy'] > self.lastvalacc: # your custom condition
            self.model.save('model_CNN_v1.h5', overwrite=True)  # or model_Inception_v1.h5
            self.lastvalacc = logs['val_accuracy']
            print('best model = ' + str(logs['val_accuracy']), file=self.log)
