from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import regularizers, initializers
from keras.layers import concatenate
from CustomModelCheckPoint import CustomModelCheckpoint
import tensorflow as tf


class InceptionV4Model(object):

    def conv2d_bn(self, x, nb_filter, nb_row, nb_col, padding='same', strides=(1,1), use_bias=False):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        x = Convolution2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=use_bias)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x


    #Inception block A
    def block_inception_a(self,input):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        branch_0 = self.conv2d_bn(input, 96, 1, 1) #<, 5, 5, 96>    

        branch_1 = self.conv2d_bn(input, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3) #<?, 5, 5, 96>    

        branch_2 = self.conv2d_bn(input, 64, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3) #<?, 5, 5, 96>    

        branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
        branch_3 = self.conv2d_bn(branch_3, 96, 1,1) #<?, 5, 5, 96>    

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis) #<?, 5, 5, 384>
        return x


    #Reduction block A
    def block_reduction_a(self,input):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.conv2d_bn(input, 384, 3, 3, strides=(2,2), padding='valid') #<?, 2, 2, 384>        

        branch_1 = self.conv2d_bn(input, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 3, 3)    
        branch_1 = self.conv2d_bn(branch_1, 256, 3, 3, strides=(2,2), padding='valid') #<?, 2, 2, 256>    
    
        branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input) #<?, 2, 2, 384    

        x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis) #<?, 2, 2, 1024>    
        return x


    #Inception block B
    def block_inception_b(self, input):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.conv2d_bn(input, 384, 1, 1) #<, 2, 2, 384>    

        branch_1 = self.conv2d_bn(input, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 256, 7, 1) #<, 2, 2,256>    
    
        branch_2 = self.conv2d_bn(input, 192, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 192, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 224, 1, 7)
        branch_2 = self.conv2d_bn(branch_2, 224, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 256, 1, 7) #<, 2, 2, 256>    
    
        branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
        branch_3 = self.conv2d_bn(branch_3, 128, 1, 1) #<, 2, 2, 128>    
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis) #<?, 2, 2, 1024>    
        return x
    

    #Reduction block B
    def block_reduction_b(self, input):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.conv2d_bn(input, 192, 1, 1)    
        branch_0 = self.conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='same') #<?, 1, 1, 192>
    
        branch_1 = self.conv2d_bn(input, 256, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 256, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 320, 7, 1)    
        branch_1 = self.conv2d_bn(branch_1, 320, 3, 3, strides=(2,2), padding='same') #<?, 1, 1, 320>
        
        branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input) #<?, 1, 1, 1024>
        
        x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis) #<?, 1, 1, 1536>
        return x


    #Inception block C
    def block_inception_c(self, input):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.conv2d_bn(input, 256, 1, 1) #<, 1, 1,256>    
    
        branch_1 = self.conv2d_bn(input, 384, 1, 1)
        branch_10 = self.conv2d_bn(branch_1, 256, 1, 3)
        branch_11 = self.conv2d_bn(branch_1, 256, 3, 1)    
        branch_1 = concatenate([branch_10, branch_11], axis=channel_axis) #<, 1, 1, 512>    

        branch_2 = self.conv2d_bn(input, 384, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 448, 3, 1)
        branch_2 = self.conv2d_bn(branch_2, 512, 1, 3)
        branch_20 = self.conv2d_bn(branch_2, 256, 1, 3)
        branch_21 = self.conv2d_bn(branch_2, 256, 3, 1)    
        branch_2 = concatenate([branch_20, branch_21], axis=channel_axis) #<, 1, 1, 512>    

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self.conv2d_bn(branch_3, 256, 1, 1) #<, 1, 1,256>    
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis) #<?, 1, 1, 1536>    
        return x


    #Inception stem
    def inception_v4_base(self, input):
        #print("input.shape = " + str(input.shape)) #<?, 64, 64, 1>
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1     

        # Input Shape is 64 x 64 x 1
    
        net = self.conv2d_bn(input, 32, 3, 3, strides=(2,2), padding='valid') #<, 31, 31, 32>    
        net = self.conv2d_bn(net, 32, 3, 3, padding='valid') #<, 29, 29, 32>    
        net = self.conv2d_bn(net, 64, 3, 3) #<, 29, 29, 64>    
    
        branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net) #<, 14, 14, 64>    
        branch_1 = self.conv2d_bn(net, 96, 3, 3, strides=(2,2), padding='valid') #<, 14, 14, 96>    
        net = concatenate([branch_0, branch_1], axis=channel_axis) #<, 14, 14, 160>        

        branch_0 = self.conv2d_bn(net, 64, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, padding='valid') #<, 12, 12, 96>       
        branch_1 = self.conv2d_bn(net, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, padding='valid') #<, 12, 12, 96>    
        net = concatenate([branch_0, branch_1], axis=channel_axis) #<, 12, 12, 192>    

        branch_0 = self.conv2d_bn(net, 192, 3, 3, strides=(2,2), padding='valid') #<, 5, 5, 192>    
        branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net) #<, 5, 5, 192>    
        net = concatenate([branch_0, branch_1], axis=channel_axis) #<, 5, 5, 384>       
    
        #print("\n" + "calling Inception-A block")    
        # 5 x 5 x 384
        # 4 x Inception-A blocks    
        for idx in range(4):
            net = self.block_inception_a(net)
    
        #print("\n" + "calling Reduction-A block")        
        # 5 x 5 x 384
        # Reduction-A block
        net = self.block_reduction_a(net)
    
        #print("\n" + "calling Inception-B block")        
        # 2 x 2 x 1024
        # 7 x Inception-B blocks
        for idx in range(7):
            net = self.block_inception_b(net)
    
        #print("\n" + "calling Reduction-B block")            
        #2 x 2 x 1024
        # Reduction-B block
        net = self.block_reduction_b(net)
    
        #print("\n" + "calling Inception-C block")            
        #1 x 1 x 1536
        # 3 x Inception-C blocks
        for idx in range(3):
            net = self.block_inception_c(net)
    
        return net


    def inception_v4_model(self, img_rows, img_cols, channel, num_classes):
        if K.image_data_format() == 'channels_first':
            inputs = Input((1, 64, 64))    
            channel_axis = 1
        else:
            inputs = Input((64, 64, 1))        
            channel_axis = -1
    
        #print("K.image_dim_ordering() = " + K.image_dim_ordering()) #tf
        #print("K.image_data_format()  = " + K.image_data_format()) #channels_last    
        #print("channel_axis = " + str(channel_axis)) #-1

        # Define the input as a tensor with shape input_shape
        inputs = Input((64, 64, 1))
        #print("inputs.shape = " + str(inputs.shape)) #<?, 64, 64, 1>
    
        # Make inception base
        net = self.inception_v4_base(inputs)
    
        #print("\n" + "after inception_v4_base(), net.shape = " + str(net.shape)) #<?, 1, 1, 1536>     
    
        X = AveragePooling2D((2,2), padding='same')(net)    
        #print("after AveragePooling2D, X.shape = " + str(X.shape)) #<?, 1, 1, 1536>  
        #X = Dropout(0.2)(X)  # temp, put it back 
        #print("after Dropout, X.shape = " + str(X.shape)) #<?, 1, 1, 1536>  
        X = Flatten()(X)
        #print("X.shape = " + str(X.shape)) #<?, ?>
        #X = Dense(output_dim=num_classes, activation='softmax')(X)    
        #X = Dense(units=2, activation="softmax")(X)   
        X = Dense(units=64, activation='relu')(X) 
        X = Dense(units=1, activation='sigmoid')(X)
        #print("X.shape = " + str(X.shape)) #<?, 2>
    
        #Create model
        #print("\n" + "creating model")

        model = Model(inputs, X, name='inception_v4')
        return model

    def create_model(self, ):
        num_classes = 1    
        # input image dimensions 
        img_rows, img_cols = 64, 64
        channels = 1

        model = self.inception_v4_model(img_rows, img_cols, channels, num_classes) #building the model's graph
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #configuring the learning process by compiling the model        
        model.compile(optimizer= 'adam' , loss= tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def train_model(self, model, train_images,train_labels,test_images, test_labels, epochs):
        cbk = CustomModelCheckpoint()  # so that we can save the best model
        history = model.fit(train_images, train_labels, epochs=epochs, callbacks=[cbk], 
                        validation_data=(test_images, test_labels))
        return model

    def load_model(self,model_file_name):
        model = tf.keras.models.load_model(model_file_name)
        return model

    def evaluate(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images,  test_labels)
        return test_loss, test_acc