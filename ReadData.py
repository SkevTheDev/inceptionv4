from PIL import Image
import numpy as np
from numpy import newaxis
import os
from keras import backend as K

class ReadData(object):
    def read_data(self,clientdir, imposterdir, client_train_normalized_file, imposter_train_normalized_file):
        #-------------read train file----------------
        filepath1 = client_train_normalized_file
        filepath2 = imposter_train_normalized_file
        data = []
        label = []
        with open(filepath1) as fp1:
           line = fp1.readline()
           cnt = 1
           while line:
              # read bmp file
              imgfilepath = clientdir + line.strip().replace('\\','/')
              #print(imgfilepath)
               
              im = Image.open(imgfilepath)
              imgdata = np.array(im)
              data.append(imgdata)
              label.append(0)  # client or true person has class label of 1
              line = fp1.readline()
              cnt += 1
        with open (filepath2) as fp2:
           line = fp2.readline()
           while line:
              # read bmp file
              imgfilepath = imposterdir + line.strip().replace('\\','/')
              im = Image.open(imgfilepath)
              imgdata = np.array(im)
              data.append(imgdata)
              label.append(1)  # imposter has class label of 0
              line = fp2.readline()
              cnt += 1
        data_tensor = np.asarray(data).reshape((len(data),64,64,1))
        label_tensor = np.asarray(label).reshape((len(label),1))
        data_tensor = data_tensor/ 255.0
        return data_tensor, label_tensor

 